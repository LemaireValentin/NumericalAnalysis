import numpy as np

# The function mysolve(A, b) is invoked by ndt.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ndt.py


SolverType = 'GMRES'
rtol = 1e-8
prec = True


def mysolve(A, b):
    if SolverType == 'numpy':
        return True, np.linalg.solve(A, b)
    elif SolverType == 'GMRES':
        sA, iA, jA = CSRformat(A)
        x, res = csrGMRES(sA, iA, jA, b, rtol, prec)
        return True, x
    else:
        return False, 0


def CSRformat(A):
    """
    Translates the matrix A in the CSR format.
    @:param A: 2D numpy array (or matrix) to convert in CSR format
    @:return: a tuple containing 3 1D numpy arrays:
        sA[s] contains the sth non-zero element of A
        iA[i] contains the index in sA containing the first non-zero element of the line i in A
        jA[j] contains the index of the column containing the jth non-zero element of A
    """
    A = np.array(A)
    idx = np.nonzero(A)     # Gets the indices of non-zero elements in A
    sA = A[idx]             # Storing non-zero elements in sA
    jA = idx[1]             # Storing the columns of the non-zero elements

    iA = np.zeros(len(A)+1, dtype=int)
    for i in range(1, len(iA)):
        # Counting the number of non-zero elements in the ith line
        iA[i] = iA[i-1] + np.sum(idx[0][iA[i-1]:] == i-1)

    return sA, iA, jA


def compute_bands(iA, jA):
    """
    This functions computes the upper and lower bands of the matrix.
    This is the slow version used to evaluate complexities.
    @:param iA, jA: the indices vectors of a matrix in CSR format
                 (the value vector sA is not needed and therefore not an argument of the function)
    @:return: The lower (max_band_l) and upper (max_band_u) bands of the matrix
    """
    N = len(iA) - 1
    return np.max(np.arange(N) - jA[iA[:N]]), np.max(jA[iA[1:] - 1] - np.arange(N))


def QRsolve(A, b):
    """
    This function implements a solver for the QR decompostion. It solves the system :
        R x = Q^* b
    knowing that R is an upper triangular matrix.
    """
    Q, R = np.linalg.qr(A)
    M, N = np.shape(Q)
    y = np.dot(b, Q.conjugate())
    x = np.zeros(N, dtype=complex)
    for i in range(N - 1, -1, -1):
        x[i:i + 1] = (y[i] - np.dot(R[i, i:], x[i:])) / R[i, i]
    return x


def csrILU0(sA, iA, jA):
    """
    This function computes the ILU(0) decomposition of the CSR matrix A represented by sA, iA and jA and returns a
    CSR format matrix representative of L and U (lower and upper triangular matrices)
    @:param sA, iA, jA: 3 1D numpy arrays representing matrix A in CSR format
    @:return: 3 1D numpy arrays representing ILU(0) decomposition of A in CSR format
    """

    N = len(iA) - 1

    band_l, band_u = compute_bands(iA, jA)

    sILU = sA.copy()
    iILU = iA.copy()
    jILU = jA.copy()

    for i in range(N):
        a_ii = sILU[iILU[i] + np.where(jILU[iILU[i]:iILU[i+1]] == i)[0]]
        if abs(a_ii) == 0:
            return None, None, None

        for j in range(i+1, min(i+1+band_l, N)):
                idx = np.where(jILU[iILU[j]:iILU[j + 1]] == i)[0]
                if idx.size > 0:
                    ji_idx = iILU[j] + idx[0]
                    sILU[ji_idx] /= a_ii
                    for k in range(iILU[i], iILU[i+1]):
                        if i+1 <= jILU[k] < min(i + 1 + band_u, N):
                            idx2 = np.where(jILU[iILU[j]:iILU[j+1]] == jILU[k])[0]
                            if idx2.size > 0:
                                idx_to_change = iILU[j] + idx2[0]
                                sILU[idx_to_change] -= sILU[ji_idx] * sILU[k]
    return sILU, iILU, jILU


def csrLUsolve(sLU, iLU, jLU, b):
    """
    Solves the two triangular systems Ly = b and Ux = y in sparse format and returns solution array x
    @:param sLU, iLU, jLU: 3 numpy 1D arrays representing a LU decomposition in CSR format
                         of a matrix of dimension len(b) x len(b)
    @:param b: numpy 1D array, right member of the linear system to solve : LUx = b
    @:return: numpy 1D array representing the solution of the linear system
    """
    b = np.array(b)
    N = len(b)

    # Solves Lower triangular system Ly = b
    y = np.zeros(N, dtype=complex)
    for i in range(N):
        # Only does the scalar product for non-zero elements of L
        idx = iLU[i] + np.where(jLU[iLU[i]:iLU[i+1]] < i+1)[0]
        y[i] = b[i] - np.dot(sLU[idx].conj(), y[jLU[idx]])

    # Solves Upper triangular system Ux = y
    x = np.zeros(N, dtype=complex)
    for i in range(N - 1, -1, -1):
        # Only does the scalar product for non-zero elements of U
        idx = iLU[i] + np.where(jLU[iLU[i]:iLU[i + 1]] > i)[0]
        x[i] = (y[i] - np.dot(sLU[idx].conj(), x[jLU[idx]])) / (sLU[iLU[i] + np.where(jLU[iLU[i]:iLU[i+1]] == i)[0]])
    return x


def csrMult(sA, iA, jA, v):
    """
    Performs the matrix, vector dot product between the matrix A represented by sA, iA and jA and the vector v
        In full : returns A @ v
    @:param sA, iA, jA: 3 1D numpy arrays representing a matrix in CSR format
    @:param v: 1D numpy array on which to perform dot product.
    @:return: a 1D numpy array representing the dot product of A and v
    """
    N = len(iA) - 1
    res = np.zeros(N, dtype=complex)
    for i in range(N):
        res[i] = np.dot(sA[iA[i]:iA[i+1]], v[jA[iA[i]:iA[i+1]]])
    return res


def csrGMRES(sA, iA, jA, b, rtol, prec, max_iter=300):
    """
    Applies the GMRES algorithm (as described in report) in CSR format on the CSR matrix A represented by sA, iA and jA
    and vector b. It returns an approximation of the solution u : Au = b
    @:param sA, iA, jA: 3 1D numpy arrays representing a matrix in CSR format
    @:param b: 1D numpy array of same dimension as matrix A (len(b) == len(iA) - 1)
    @:param rtol: float scalar representing the convergence criteria
    @:prec: boolean. If true, preconditionning using ILU(0) is applied else, no preconditionning.
    @:max_iter: integer. Limiting the number of iterations the algorithm. Default is 300.
    """
    m = 0
    V = []
    H = np.zeros((max_iter+1, max_iter), dtype=complex)

    if prec:
        sILU, iILU, jILU = csrILU0(sA, iA, jA)      # Computing ILU(0) decomposition
        r = csrLUsolve(sILU, iILU, jILU, b)         # Initial residue
        beta = np.linalg.norm(r)
        V.append(r / beta)                          # First vector in base

    else:
        beta = np.linalg.norm(b)
        V.append(b / beta)                          # First vector in base

    res = [beta]

    while m < max_iter:
        # Arnoldi iteration
        if prec:
            w = csrLUsolve(sILU, iILU, jILU, csrMult(sA, iA, jA, V[m]))
        else:
            w = csrMult(sA, iA, jA, V[m])

        for i in range(m+1):
            H[i, m] = np.dot(V[i].conj(), w)
            w -= H[i][m] * V[i]

        H[m+1, m] = np.linalg.norm(w)
        V.append(w / H[m+1, m])

        vec = np.zeros(m+2, dtype=complex)
        vec[0] = beta

        # Finding y to minimise residue
        y = QRsolve(H[:m+2, :m+1], vec)
        newres = np.linalg.norm(np.dot(H[:m+2, :m+1], y) - vec)
        res.append(newres)

        if newres < rtol:                           # If the algorithm has converged, we stop it
            m += 1
            break
        m += 1

    # Computing solution
    u = np.dot(y, V[:m])

    return u, np.array(res)

