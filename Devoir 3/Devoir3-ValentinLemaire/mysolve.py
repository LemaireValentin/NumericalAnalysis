import numpy as np

# The function mysolve(A, b) is invoked by ndt.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ndt.py

SolverType = 'LUcsr-rcmk'

tol = 1e-15

def mysolve(A, b):
    if SolverType == 'numpy':
        return True, np.linalg.solve(np.array(A), np.array(b))
    elif SolverType == 'LU':
        LUres, P = LU(A)
        return True, LUsolve(LUres, b, P)
    elif SolverType == 'LUcsr-rcmk':
        sA, iA, jA = CSRformat(A)
        r = RCMK(iA, jA)
        r_inv = invert_r(r)
        sA, iA, jA = reduce_bands(sA, iA, jA, r, r_inv)
        sLU, iLU, jLU = LUcsr(sA, iA, jA)
        return True, LUsolve_csr(sLU, iLU, jLU, b[r])[r_inv]
    elif SolverType == 'LUcsr':
        sA, iA, jA = CSRformat(A)
        sLU, iLU, jLU = LUcsr(sA, iA, jA)
        return True, LUsolve_csr(sLU, iLU, jLU, b)
    elif SolverType == 'GMRES':
        return False, 0
    else:
        return False, 0


# ============= CSR FUNCTIONS =============

def CSRformat(A):
    """"
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
    :param iA, jA: the indices vectors of a matrix in CSR format
                 (the value vector sA is not needed and therefore not an argument of the function)
    :return: The lower (max_band_l) and upper (max_band_l) bands of the matrix
    """
    N = len(iA) - 1
    return np.max(np.arange(N) - jA[iA[:N]]), np.max(jA[iA[1:] - 1] - np.arange(N))


def create_fill_in(sA, iA, jA, band_l, band_r):
    """
    This function creates 3 new vectors : sLU, iLU, jLU which are similar to sA, iA, jA but contains more elements.
    Indeed, since the LU algorithm only modifies the elements inside the matrix's band, this function creates the
    arrays sLU, iLU, jLU in a way that they could contain every element inside the matrix's band.
    sLU contains non-zero elements for the corresponding elements in sA.
    :param sA, iA, jA : 3 numpy 1D arrays representing a matrix in CSR format.
    :param band_l, band_r : upper and lower bands of the matrix represented by (sA, iA, jA),
                            result of the compute_bands(iA, jA, N) function
    :return: 3 numpy 1D arrays : (sLU, iLU, jLU) representing the same matrix (sA, iA, jA) but in the format defined above
    """
    N = len(iA) - 1

    # Computes the number of elements inside the matrix's band
    num_elems = int((band_l + band_r + 1) * N - band_l * (band_l + 1)/2 - band_r * (band_r + 1)/2)

    sLU = np.zeros(num_elems, dtype=complex)
    iLU = np.zeros(len(iA), dtype=int)
    jLU = np.zeros(num_elems, dtype=int)

    for i in range(1, len(iLU)):
        # Updates iLU[i] with the sum of iLU[i-1] and the number of elements in this line
        iLU[i] = iLU[i - 1] + band_l + band_r + 1 + min(i - 1 - band_l, 0) + min(N - i - band_l, 0)
        # Creates a vector containing the indices of the column of this line's elements
        jLU[iLU[i - 1]:iLU[i]] = np.arange(i - band_l - min(i - 1 - band_l, 0) - 1, i + band_r + min(N - i - band_r, 0))
        # Adds the elements of sA to sLU
        idx = jA[iA[i - 1]:iA[i]]
        sLU[iLU[i - 1]:iLU[i]][idx - jLU[iLU[i - 1]]] = sA[iA[i - 1]:iA[i]]

    return sLU, iLU, jLU


def remove_zeros(sLU, iLU, jLU):
    """
    This function removes the elements of sLU that are still non-zero after the LU algorithm.
    It also updates iLU and jLU accordingly.
    :param sLU, iLU, jLU : A matrix represented in the format defined in the create_fill_in function
    :return: (sLU, iLU, jLU) the same matrix represented by the parameters (sLU, iLU, jLU) but in CSR format.
    """
    N = len(iLU) - 1
    nz_idx = np.nonzero(sLU)        # Gets the indices of the non-zero elements of sLU
    sLU = sLU[nz_idx]               # We only keep non-zero elements in sLU
    jLU = jLU[nz_idx]               # We only keep the column indices of non-zero elements

    for i in range(1, N + 1):
        # Updates iLU[i] by counting the number of non-zero in the lines < i
        iLU[i] = np.sum(nz_idx < iLU[i])
    return sLU, iLU, jLU


def LUcsr(sA, iA, jA):
    """
    This function performs the LU algorithm with a sparse matrix and returns a sparse matrix in the form of 3 arrays
    but was vectorized

    :param sA, iA, jA: 3 1D numpy arrays representing a 2D matrix in CSR format
    :return: sLU, iLU, jLU: 3 1D numpy arrays representing the LU decomposition
    (of the matrix represented by the parameters) in CSR format
    """
    N = len(iA) - 1

    # Computes the band and the possible fill-in for this matrix
    band_l, band_r = compute_bands(iA, jA)
    sLU, iLU, jLU = create_fill_in(sA, iA, jA, band_l, band_r)

    for i in range(N):
        a_ii = sLU[iLU[i] + i - jLU[iLU[i]]]
        if abs(a_ii) == 0:
            return None, None, None
        # Vectorized operations to divide column (inside band) by LU[i, i]
        # and update the sub-matrix (again, inside the band)

        # Computing the indices of where we need to modify the sparse matrix
        j_max = i + band_l + min(N - i - band_l, 1)
        k_max = i + band_r + 1 + min(N - i - band_r, 0)
        column_indices = iLU[i + 1] + np.where(jLU[iLU[i+1]:iLU[j_max]] == i)[0]
        line_indices = iLU[i] + np.where(np.logical_and(i + 1 <= jLU[iLU[i]:iLU[i + 1]], jLU[iLU[i]:iLU[i + 1]] < k_max))[0]
        sub_matrix_indices = iLU[i + 1] + np.where(np.logical_and(i + 1 <= jLU[iLU[i + 1]:iLU[j_max]], jLU[iLU[i + 1]:iLU[j_max]] < k_max))[0]

        # Dividing the column by LU[i, i] (only inside the band)
        sLU[column_indices] /= a_ii

        # Updates the sub-matrix (only the elements that will change)
        sLU[sub_matrix_indices] -= np.outer(sLU[column_indices], sLU[line_indices]).ravel()

    # Removes the remaining zeros and returns the sparse matrix representing LU
    return remove_zeros(sLU, iLU, jLU)


def LUsolve_csr(sLU, iLU, jLU, b):
    """
    Solves the two triangular systems Ly = b and Ux = y in sparse format and returns solution array x
    :param sLU, iLU, jLU: 3 numpy 1D arrays representing a LU decomposition in CSR format
                         of a matrix of dimension len(b) x len(b)
    :param b: numpy 1D array, right member of the linear system to solve : LUx = b
    :return: numpy 1D array representing the solution of the linear system
    """
    b = np.array(b)
    N = len(b)

    # Solves Lower triangular system Ly = b
    y = np.zeros(N, dtype=complex)
    for i in range(N):
        # Only does the scalar product for non-zero elements of L
        idx = iLU[i] + np.where(jLU[iLU[i]:iLU[i+1]] < i+1)[0]
        y[i] = b[i] - np.dot(sLU[idx], y[jLU[idx]])

    # Solves Upper triangular system Ux = y
    x = np.zeros(N, dtype=complex)
    for i in range(N - 1, -1, -1):
        # Only does the scalar product for non-zero elements of U
        idx = iLU[i] + np.where(jLU[iLU[i]:iLU[i + 1]] > i)[0]
        x[i] = (y[i] - np.dot(sLU[idx], x[jLU[idx]])) / (sLU[iLU[i] + np.where(jLU[iLU[i]:iLU[i+1]] == i)[0]])
    return x


# ============= RCMK FUNCTIONS =============


def RCMK(iA, jA):
    """
    This function computes the permutation vector r of the sparse matrix by applying the RCMK algorithm
    :param iA, jA: the two last vectors of the CSR format of a matrix (sA is not needed)
    :return: numpy 1D array representing the permutation vector r, the solution of the RCMK algorithm.
    """

    N = len(iA) - 1
    r = np.array([-1] * N)              # Permutation vector r
    q = np.array([-1] * N)              # Queue to store the nodes (max length of queue = N)
    r_i, q_i, q_j = N-1, 0, 0           # Indices to navigate in r and q

    # Array of booleans, not_in_q[i] = True if the node i hasn't been added to q before, False otherwise
    not_in_q = np.array([True] * N)

    degree = iA[1:] - iA[:N]            # Stores the degree of each node represented by adjacency matrix A

    while r[0] == -1:                   # As long as there are nodes that haven't been added to r
        if q[q_i] == -1:                # If the queue is empty
            c = np.argmin(degree)       # We get the lowest degree node
            not_in_q[c] = False
            degree[c] = N+1             # We set the degree of this node to N+1 to avoid putting it twice in r
        else:                           # If the queue is not empty
            c = q[q_i]                  # We "pop" the first element of the queue
            q_i += 1

        # We get the indices of the neighbors of c in increasing order of degree
        # (We only take the ones that haven't been added to the queue already)
        to_append = sorted(jA[iA[c]:iA[c + 1]][not_in_q[jA[iA[c]:iA[c + 1]]]], key=lambda x: degree[x])

        # We add those nodes to the queue
        q[q_j:q_j+len(to_append)] = to_append
        q_j += len(to_append)
        not_in_q[to_append] = False
        degree[to_append] = N + 1

        # We add c to r
        r[r_i] = c
        r_i -= 1

    return r


def invert_r(r):
    """
    This function computes the inverse permutation vector r_inv. A[r[r_inv]] = A[r_inv[r]] = A
    @:param r: numpy 1D array representing the permutation vector obtained by RCMK(iA, jA)
    @:return: numpy 1D array representing the inverse permutation vector
    """
    r_inv = np.zeros(len(r), dtype=int)
    r_inv[r] = np.arange(len(r))
    return r_inv


def reduce_bands(sA, iA, jA, r, r_inv):
    """
    This function applies the permutation vector to the sparse matrix.
    Hence, in a full matrix, A[i, j] would become A[r[i], r[j]]
    :param sA, iA, jA: 3 numpy 1D arrays representing a matrix A in CSR format
    :param r: 1D numpy array representing the permutation vector obtained by RCMK(iA, jA)
    :param r_inv: 1D numpy array representing the inverse permutation vector obtained by invert_r(r)
    :return: 3 numpy 1D arrays representing a CSR format matrix in which
             the permutation was applied (as described above)
    """
    N = len(iA) - 1
    M = len(jA)

    # We initialize new sparse matrix arrays
    sA2, iA2, jA2 = np.zeros(M, dtype=complex), np.zeros(N+1, dtype=int), np.zeros(M, dtype=int)

    for i in range(1, N+1):
        # We initialize iA2[i] to iA2[i-1] + the length of the ith line in the permuted matrix
        iA2[i] = iA2[i - 1] + iA[r[i-1]+1] - iA[r[i-1]]

        # We add the columns of the elements of the ith line in the permuted matrix to jA2
        jA2[iA2[i-1]:iA2[i]] = r_inv[jA[iA[r[i-1]]:iA[r[i-1]+1]]]

        # We get the indices array that will sort the column indices of the ith line in the permuted matrix
        ind = np.argsort(jA2[iA2[i-1]:iA2[i]], kind='mergesort')

        # We sort the column indices of the ith line in the permuted matrix
        jA2[iA2[i - 1]:iA2[i]] = jA2[iA2[i-1]:iA2[i]][ind]

        # We sort the values of the ith line in the permuted matrix according to the order of the column indices
        sA2[iA2[i-1]:iA2[i]] = sA[iA[r[i-1]]:iA[r[i-1]+1]][ind]

    return sA2, iA2, jA2


# ============= FULL MATRICES FUNCTIONS =============


def LU(A):
    """
    Fast implementation of LU, implements the LU decomposition algorithm
    WARNING : this algorithm is in-place and hence changes the values in A.
    :param A: Numpy array (or matrix) on which the LU decomposition will be made
    :return: The matrix representing the LU decomposition (L and U combined) and a permutation vector P
    """

    A = np.array(A)
    N = len(A)
    P = np.arange(N + 1)
    P[N] = 0
    for i in range(N):
        imax = i + np.argmax(np.abs(A[P[i:N], i]))
        if abs(A[P[imax], i]) <= tol:
            return None, None

        if imax != i:
            P[i], P[imax] = P[imax], P[i]
            P[N] += 1

        A[P[i+1:N], i:i+1] /= A[P[i], i]
        A[P[i+1:N], i+1:] -= np.outer(A[P[i+1:N], i], A[P[i], i+1:])
    return A, P


def LUsolve(LU, b, P):
    """
    This function solves the linear system : LUx = Pb by solving two consecutive systems:
            Ly = Pb
            Ux = y
        knowing that L and U are lower and upper triangular matrices respectively
    :param LU: Numpy 2D array representing the LU decomposition of a coefficient matrix A (L and U combined, result of LU(A))
    :param b: Numpy 1D array representing the independt terms of the linear system : Ax = b
    :param P: Numpy 1D array representing the permutation vector (result of LU(A))
    :return: the solution to the linear system Ax = b and LUx = Pb (both are equivalent)
    """
    """
        This function solves the linear system : LUx = Pb by solving two consecutive systems:
            Ly = Pb
            Ux = y
        knowing that L and U are lower and upper triangular matrices respectively
    """
    N = len(LU)
    y = np.zeros(N, dtype=complex)
    for i in range(N):
        y[i:i+1] = (b[P[i]] - np.dot(LU[P[i], :i+1], y[:i+1]))
    x = np.zeros(N, dtype=complex)
    for i in range(N-1, -1, -1):
        x[i:i+1] = (y[i] - np.dot(LU[P[i], i:], x[i:])) / LU[P[i], i]
    return x


# =====================================================================================================
# The following functions were only written and used to evaluate complexities and compare performances.
# They should not be used. Prefer the functions above.
# =====================================================================================================


# ============== SLOW CSR FUNCTIONS ==============


def CSRformat_slow(A):
    """"
        Translates the matrix A in the CSR format. This is the slow version used to evaluate complexities
        @:param A: 2D numpy array (or matrix) to convert in CSR format
        @:return: a tuple containing 3 1D numpy arrays:
            sA[s] contains the sth non-zero element of A
            iA[i] contains the index in sA containing the first non-zero element of the line i in A
            jA[j] contains the index of the column containing the jth non-zero element of A
    """

    A = np.array(A)
    N = len(A)
    M = 0     # Counter for the number of nonzero elements in A

    for i in range(N):
        for j in range(N):
            if abs(A[i, j]) != 0:
                M += 1

    # Initializing the vectors with zeros
    sA = []
    iA = np.zeros(len(A)+1, dtype=int)
    jA = []

    for i in range(N):
        count_in_line = 0      # To keep track of the number of non-zero elements in this line
        for j in range(N):
            if abs(A[i, j]) > tol:  # If the element is non-zero we add it to the CSR format of A
                sA.append(A[i, j])
                jA.append(j)
                count_in_line += 1
        iA[i+1] = iA[i] + count_in_line

    return np.array(sA, dtype=complex), iA, np.array(jA, dtype=int)


def compute_bands_slow(iA, jA):
    """
    This functions computes the upper and lower bands of the matrix.
    This is the slow version used to evaluate complexities.
    :param iA, jA: the indices vectors of a matrix in CSR format
                 (the value vector sA is not needed and therefore not an argument of the function)
    :param N: Number of lines in the matrix that represented by the CSR format (N == len(iA) - 1)
    :return: The lower (max_band_l) and upper (max_band_l) bands of the matrix
    """

    N = len(iA) - 1
    max_band_l, max_band_r = 0, 0
    for i in range(N):
        # Saves the max between the previous max and the right (and left) band on this line
        max_band_l = max(max_band_l, i - jA[iA[i]])
        max_band_r = max(max_band_r, jA[iA[i+1]-1] - i)
    return max_band_l, max_band_r


def remove_zeros_slow(sLU, iLU, jLU):
    """
    This function removes the elements of sLU that are still non-zero after the LU algorithm.
    It also updates iLU and jLU accordingly. This is a slow version used to evaluate complexities
    :param sLU, iLU, jLU : A matrix represented in the format defined in the create_fill_in function
    :return: (sLU, iLU, jLU) the same matrix represented by the parameters (sLU, iLU, jLU) but in CSR format.
    """

    N = len(iLU) - 1

    # Initializes new vectors
    new_sLU = []
    new_jLU = []
    new_iLU = np.zeros(N + 1, dtype=int)

    for i in range(1, N + 1):
        count_in_line = 0                   # Keeps track of the non-zero elements in this line
        for j in range(iLU[i - 1], iLU[i]):
            if abs(sLU[j]) > tol:            # If this element is non-zero we add it to the new vectors
                new_sLU.append(sLU[j])
                new_jLU.append(jLU[j])
                count_in_line += 1
        new_iLU[i] = new_iLU[i-1] + count_in_line
    return np.array(new_sLU, dtype=complex), new_iLU, np.array(new_jLU, dtype=int)


def LUcsr_slow(sA, iA, jA):
    """
    This function performs the LU algorithm with a sparse matrix and returns a sparse matrix in the form of 3 arrays
    This is a slow version used to evaluate complexities.

    :param sA, iA, jA: 3 1D numpy arrays representing a 2D matrix in CSR format
    :return: sLU, iLU, jLU: 3 1D numpy arrays representing the LU decomposition
    (of the matrix represented by the parameters) in CSR format
    """

    N = len(iA) - 1

    # Computes the band and the possible fill-in for this matrix
    band_l, band_r = compute_bands_slow(iA, jA)
    sLU, iLU, jLU = create_fill_in(sA, iA, jA, band_l, band_r)

    for i in range(N):
        # Gets LU[i, i] in the sparse matrix
        a_ii = sLU[iLU[i] + i - jLU[iLU[i]]]

        if abs(a_ii) == 0:
            return None, None, None

        # Gets the LU[i, i+1:] line (but only the elements inside the band)
        idx_line = range(i+1, i + band_r + min(N - i - band_r, 1))
        line = [sLU[iLU[i] + k - jLU[iLU[i]]] for k in idx_line]

        for j in range(i + 1, i + band_l + min(N - i - band_l, 1)):
            # Divides the column below (i, i) by A[i, i] (only inside the band)
            sLU[iLU[j] + i - jLU[iLU[j]]] /= a_ii
            # Gets LU[j, i] in the sparse matrix
            a_ji = sLU[iLU[j] + i - jLU[iLU[j]]]

            for k in range(len(line)):
                # Updates LU[j, k] (only the elements that will change)
                sLU[iLU[j] + idx_line[k] - jLU[iLU[j]]] -= a_ji * line[k]

    # Removes the remaining zeros and returns the sparse matrix representing LU
    return remove_zeros_slow(sLU, iLU, jLU)


def LUsolve_csr_slow(sLU, iLU, jLU, b):
    """
    Solves the two triangular systems Ly = b and Ux = y in sparse format and returns solution array x
    This is a slow version used to evaluate complexities
    :param sLU, iLU, jLU: 3 numpy 1D arrays representing a LU decomposition in CSR format
                         of a matrix of dimension len(b) x len(b)
    :param b: numpy 1D array, right member of the linear system to solve : LUx = b
    :return: numpy 1D array representing the solution of the linear system
    """
    b = np.array(b)
    N = len(b)

    # Solves Lower triangular system Ly = b
    y = np.zeros(N, dtype=complex)
    for i in range(N):
        y[i] = b[i]
        for j in range(iLU[i], iLU[i+1]):
            if jLU[j] < i:          # Only does the scalar product for non-zero elements of L
                y[i] -= sLU[j] * y[jLU[j]]

    # Solves Upper triangular system Ux = y
    x = np.zeros(N, dtype=complex)
    for i in range(N - 1, -1, -1):
        u_ii = 0
        x[i] = y[i]
        for j in range(iLU[i], iLU[i + 1]):
            if jLU[j] == i:
                u_ii = sLU[j]
            if jLU[j] > i:          # Only does the scalar product for non-zero elements of U
                x[i] -= sLU[j] * x[jLU[j]]
        x[i] /= u_ii                # Divides by diagonal element of U
    return x


# =============== SLOW FULL MATRICES FUNCTIONS ===============

def LU_slow(A):
    """
        Simple implementation of LU algorithm with 3 for loops, this implementation is slow
        and therefore is not the one used in mysolve
    """
    A = np.array(A)
    N = len(A)
    P = np.arange(N + 1)
    P[N] = 0
    for i in range(N):
        imax = i
        max = A[P[i], i]
        for j in range(i+1, N):
            if abs(A[P[j], i]) > abs(max):
                imax = j
                max = A[P[j], i]
        if abs(A[P[imax], i]) == tol:
            return None, None
        if imax != i:
            P[i], P[imax] = P[imax], P[i]
            P[N] += 1
        for j in range(i+1, N):
            A[P[j], i] /= A[P[i], i]
            for k in range(i+1, N):
                A[P[j], k] -= A[P[j], i] * A[P[i], k]
    return A, P


def LU_no_pivot_slow(A):
    """
    This function was used to evaluate complexities and should not be used for an efficient LU decomposition.
    It implements the LU decomposition without pivoting
    :param A: Numpy array (or matrix) on which the LU decomposition will be made
    :return: The matrix representing the LU decomposition (L and U combined)
    """
    A = np.array(A)
    N = len(A)
    P = np.arange(N + 1)
    P[N] = 0
    for i in range(N):
        if np.abs(A[i, i]) == 0:
            return None, None
        for j in range(i + 1, N):
            A[j, i] /= A[i, i]
            for k in range(i + 1, N):
                A[j, k] -= A[j, i] * A[i, k]
    return A


def LU_no_pivot(A):
    """
    This function was used to compare the CSR format to LU decomposition without pivoting
    and should not be used for an efficient LU decomposition.
    It implements the LU decomposition without pivoting
    :param A: Numpy array (or matrix) on which the LU decomposition will be made
    :return: The matrix representing the LU decomposition (L and U combined)
    """
    A = np.array(A)
    N = len(A)
    for i in range(N):
        if abs(A[i, i]) <= tol:
            return None, None
        A[i:, i:i + 1] /= A[i, i]
        A[i + 1:, i + 1:] -= np.outer(A[i + 1:, i], A[i, i + 1:])
    return A
