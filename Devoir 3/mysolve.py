import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import time

# The function mysolve(A, b) is invoked by ndt.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ndt.py

# SolverType = 'LU'

def mysolve(A, b, SolverType):
    if SolverType == 'scipy':
        return True, scipy.sparse.linalg.spsolve(A, b)
    if SolverType == 'numpy':
        return True, np.linalg.solve(A, b)
    elif SolverType == 'QR':
        return True, QRsolve(np.array(A, dtype=complex), np.array(b, dtype=complex))
    elif SolverType == 'LU':
        LUres, P = LU(np.array(A))
        return True, LUsolve(LUres, b, P)
    elif SolverType == 'GMRES':
        return False, 0
    else:
        return False, 0


"""
    Simple implementation of LU algorithm with 3 for loops, this implementation is slow 
    and therefore is not the one used in mysolve
"""
def LU_slow(A):
    tol = 1e-15
    N = len(A)
    P = np.arange(N + 1)
    P[N] = 0
    for i in range(N):
        imax = i + np.argmax(np.abs(A[i:, i]))
        if abs(A[imax, i]) <= tol:
            return None, None
        if imax != i:
            P[i], P[imax] = P[imax], P[i]
            A[i], A[imax] = A[imax].copy(), A[i].copy()
            P[N] += 1
        for j in range(i+1, N):
            A[j, i] /= A[i, i]
            for k in range(i+1, N):
                A[j, k] -= A[j, i] * A[i, k]
    return A, P


"""
    Faster implementation of LU, implements the same algorithm as LU_slow(A) but was vectorised 
    and is therefore faster
    
    WARNING : the output matrix is not the same as LU_slow(A). 
              Here, you have to use A[P] to get the same matrix as the output of LU_slow(A)
"""
def LU(A):
    tol = 1e-15
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


"""
    This function solves the linear system : LUx = Pb by solving two consecutive systems:
        Ly = Pb
        Ux = y
    knowing that L and U are lower and upper triangular matrices respectively
"""
def LUsolve(A, b, P):
    N = len(A)
    y = np.zeros(N, dtype=complex)
    for i in range(N):
        y[i:i+1] = (b[P[i]] - np.dot(A[P[i], :i+1], y[:i+1]))
    x = np.zeros(N, dtype=complex)
    for i in range(N-1, -1, -1):
        x[i:i+1] = (y[i] - np.dot(A[P[i], i:], x[i:])) / A[P[i], i]
    return x


"""
    This function implements the ILU(0) algorithm. This version is slow because 
    it uses 3 for loops (see ILU0 and ILU0_always_pivot for faster versions)
"""
def ILU0_slow(A):
    tol = 1e-15
    N = len(A)
    P = np.arange(N + 1)
    P[N] = 0
    for i in range(N):
        imax = i + np.argmax(np.abs(A[i:, i]))
        if abs(A[imax, i]) <= tol:
            return None, None
        if imax != i:
            P[i], P[imax] = P[imax], P[i]
            A[i], A[imax] = A[imax].copy(), A[i].copy()
            P[N] += 1
        count = 0
        for j in range(i+1, N):
            if np.abs(A[j, i]) > tol:
                A[j, i] /= A[i, i]
                for k in range(i+1, N):
                    if np.abs(A[j, k]) > tol:
                        A[j, k] -= A[j, i] * A[i, k]
                        count +=1
    return A, P


"""
    This function implements the same ILU(0) algorithm as ILU0_slow(A) but it was vectorized to be faster. 
    Because this function ALWAYS searches for the highest pivot, but doesn't make the computation on all elements, 
    it sometimes fails to find a non-zero pivot and therefore fails. See ILU0(A) for a better version.
"""
def ILU0_always_pivot(A):
    tol = 1e-15
    N = len(A)
    P = np.arange(N + 1)
    P[N] = 0
    for i in range(N):
        imax = i + np.argmax(np.abs(A[P[i:N], i]))
        if np.abs(A[P[imax], i]) <= tol:
            imax = i
        if imax != i:
            P[i], P[imax] = P[imax], P[i]
            P[N] += 1
        idx1 = np.nonzero(np.abs(A[P[i+1:N], i]) > tol)[0]
        A[P[i + 1 + idx1], i] /= A[P[i], i]
        idx2 = np.nonzero(np.abs(A[P[i + 1:N], i + 1:]) > tol)
        A[P[i+1+idx2[0]], i+1+idx2[1]] -= A[P[i+1+idx2[0]], i] * A[P[i], i+1 + idx2[1]]
    return A, P


"""
    This function implements the same ILU(0) algorithm as the two functions above with the 
    difference that it only searches for the highest pivot in the elements in the current column A[P[i:N], i])
    if A[P[i], i] is too small to be a pivot. This function works for all tested cases.  
"""
def ILU0(A):
    tol = 1e-15
    N = len(A)
    P = np.arange(N + 1)
    P[N] = 0
    for i in range(N):
        if np.abs(A[P[i], i]) < tol:
            imax = i + np.argmax(np.abs(A[P[i:N], i]))
            if np.abs(A[P[imax], i]) <= tol:
                return None, None
        else:
            imax = i
        if imax != i:
            P[i], P[imax] = P[imax], P[i]
            P[N] += 1
        idx1 = np.nonzero(np.abs(A[P[i+1:N], i]) > tol)[0]
        A[P[i + 1 + idx1], i:i+1] /= A[P[i], i]
        idx2 = np.nonzero(np.abs(A[P[i + 1:N], i + 1:]) > tol)
        A[P[i+1+idx2[0]], i+1+idx2[1]] -= A[P[i+1+idx2[0]], i] * A[P[i], i+1+idx2[1]]
    return A, P


"""
    This function implements the QR decomposition of A in a slow manner 
"""
def QR_slow(A):
    M, N = np.shape(A)
    Q = np.zeros((M, N), dtype=complex)
    R = np.zeros((N, N), dtype=complex)
    for i in range(N):
        R[i, i] = np.linalg.norm(A[:, i], ord=2)
        Q[:, i] = A[:, i] / R[i, i]
        for j in range(i+1, N):
            sum = 0
            for k in range(i+1, N):
                sum += Q[k, i].conjugate() * A[k, j]
                A[k, j] -= R[i, j]*Q[k, j]
            R[i, j] = sum
    return Q, R


"""
    This function implements the same algorithm as the QR_slow(A) function but was fully vectorized
    It also uses A to store Q to save memory and to avoid using new matrices
"""
def QR_columns(A):
    M, N = np.shape(A)
    R = np.zeros((N, N), dtype=complex)
    for i in range(N):
        R[i, i] = np.linalg.norm(A[:, i], ord=2)
        A[:, i] = A[:, i] / R[i, i]
        R[i, i+1:] = np.dot(A[:, i].conjugate(), A[:, i+1:])
        A[:, i+1:] -= np.outer(A[:, i], R[i, i+1:])
    return A, R


"""
    This function implements the same algorithm as the two functions above but transposes A at 
    the beginning and Q at the end. 
    This way, we can use the fact that matrices are stored by rows (and not by columns) and the fact 
    that operations on rows are faster than operations on columns
    It also uses A to store Q to save memory and to avoid using new matrices
"""
def QR(A):
    A = np.array(A, dtype=complex)
    M, N = A.shape
    A2 = np.zeros(A.T.shape, dtype=complex)
    for i in range(len(A[0])):
        A2[i] = A[:, i].copy()
    A = A2
    R = np.zeros((N, N), dtype=complex)
    for i in range(N):
        R[i, i] = np.linalg.norm(A[i], ord=2)
        A[i] = A[i] / R[i, i]
        R[i, i + 1:] = np.dot(A[i + 1:, :], A[i].conjugate())
        A[i+1:, :] -= np.outer(R[i, i+1:], A[i, :])
    return A.T, R


"""
    This function implements a solver for the QR decompostion. It solves the system : 
        R x = Q^* b
    knowing that R is an upper triangular matrix.
"""
def QRsolve(A, b):
    Q, R = QR(A)
    M, N = np.shape(Q)
    y = np.dot(b, Q.conjugate())
    x = np.zeros(N, dtype=complex)
    for i in range(N - 1, -1, -1):
        x[i:i + 1] = (y[i] - np.dot(R[i, i:], x[i:])) / R[i, i]
    return x


def CSRformat(A):
    idx = np.nonzero(A)
    sA = A[idx]
    iA = np.zeros(len(A)+1, dtype=int)
    sA[:] = A[idx]
    jA = idx[1]
    for i in range(1, len(iA)):
        iA[i] = iA[i-1] + np.sum(idx[0][iA[i-1]:] == i-1)

    return sA, iA, jA


def LUcsr(sA, iA, jA):
    N = len(iA) - 1
    band_l, band_r = np.max(np.arange(N) - jA[iA[:N]]), np.max(jA[iA[1:]-1] - np.arange(N))
    num_elems = int((band_l + band_r + 1) * N - band_l * (band_l + 1)/2 - band_r * (band_r + 1)/2)
    sLU = np.zeros(num_elems, dtype=complex)
    iLU = np.zeros(len(iA), dtype=int)
    jLU = np.zeros(num_elems, dtype=int)
    for i in range(1, len(iLU)):
        iLU[i] = iLU[i-1] + band_l + band_r + 1 + min(i - 1 - band_l, 0) + min(N - i - band_l, 0)
        jLU[iLU[i-1]:iLU[i]] = np.arange(i - band_l - min(i - 1 - band_l, 0) - 1, i + band_r + min(N - i - band_r, 0))
    for i in range(N):
        idx = jA[iA[i]:iA[i+1]]
        sLU[iLU[i]:iLU[i+1]][idx - jLU[iLU[i]]] = sA[iA[i]:iA[i+1]]

    for i in range(N):
        a_ii = sLU[iLU[i] + np.where(jLU[iLU[i]:iLU[i+1]] == i)[0]]
        for j in range(i + 1, i + band_l + min(N - i - band_l, 1)):
            sLU[iLU[j] + np.where(jLU[iLU[j]:iLU[j + 1]] == i)[0]] /= a_ii
            a_ji = sLU[iLU[j] + np.where(jLU[iLU[j]:iLU[j + 1]] == i)[0]]
            for k in range(i+1, i + band_r + 1 + min(N - i - band_r, 0)):
                a_ik = sLU[iLU[i] + np.where(jLU[iLU[i]:iLU[i + 1]] == k)[0]]
                sLU[iLU[j] + np.where(jLU[iLU[j]:iLU[j+1]] == k)[0]] -= a_ji * a_ik

    nz_count = np.count_nonzero(sLU)
    new_sLU = np.zeros(nz_count, dtype=complex)
    new_jLU = np.zeros(nz_count, dtype=int)
    new_iLU = iLU.copy()

    for i in range(1, N+1):
        nz_index = np.nonzero(sLU[iLU[i-1]:iLU[i]])[0]
        new_iLU[i] = new_iLU[i-1] + len(nz_index)
        new_sLU[new_iLU[i-1]:new_iLU[i]] = sLU[iLU[i-1]:iLU[i]][nz_index]
        new_jLU[new_iLU[i-1]:new_iLU[i]] = jLU[iLU[i-1]:iLU[i]][nz_index]

    return new_sLU, new_iLU, new_jLU


def LUcsr_opt(sA, iA, jA):
    N = len(iA) - 1
    band_l, band_r = np.max(np.arange(N) - jA[iA[:N]]), np.max(jA[iA[1:]-1] - np.arange(N))
    num_elems = int((band_l + band_r + 1) * N - band_l * (band_l + 1)/2 - band_r * (band_r + 1)/2)
    sLU = np.zeros(num_elems, dtype=complex)
    iLU = np.zeros(N+1, dtype=int)
    jLU = np.zeros(num_elems, dtype=int)
    for i in range(1, N+1):
        iLU[i] = iLU[i-1] + band_l + band_r + 1 + min(i - 1 - band_l, 0) + min(N - i - band_l, 0)
        jLU[iLU[i-1]:iLU[i]] = np.arange(i - band_l - min(i - 1 - band_l, 0) - 1, i + band_r + min(N - i - band_r, 0))
        idx = jA[iA[i-1]:iA[i]]
        sLU[iLU[i-1]:iLU[i]][idx - jLU[iLU[i-1]]] = sA[iA[i-1]:iA[i]]

    for i in range(N):
        a_ii = sLU[iLU[i] + np.where(jLU[iLU[i]:iLU[i+1]] == i)[0]]
        if a_ii == 0:
            return None, None, None
        j_max = i + band_l + min(N - i - band_l, 1)
        k_max = i + band_r + 1 + min(N - i - band_r, 0)
        sLU[iLU[i + 1] + np.where(jLU[iLU[i+1]:iLU[j_max]] == i)[0]] /= a_ii
        sLU[iLU[i+1] + np.where(np.logical_and(i + 1 <= jLU[iLU[i+1]:iLU[j_max]], jLU[iLU[i+1]:iLU[j_max]] < k_max))[0]] -= np.outer(sLU[iLU[i + 1] + np.where(jLU[iLU[i+1]:iLU[j_max]] == i)[0]], sLU[iLU[i] + np.where(np.logical_and(i + 1 <= jLU[iLU[i]:iLU[i + 1]], jLU[iLU[i]:iLU[i + 1]] < k_max))[0]]).flatten()

    nz_idx = np.nonzero(sLU)
    sLU = sLU[nz_idx]
    jLU = jLU[nz_idx]

    for i in range(1, N+1):
        iLU[i] = np.sum(nz_idx < iLU[i])

    return sLU, iLU, jLU


def RMCK(iA, jA):
    pass
