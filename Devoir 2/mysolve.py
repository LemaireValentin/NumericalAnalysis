import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import time

# The function mysolve(A, b) is invoked by ndt.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ndt.py

SolverType = 'QR'

def mysolve(A, b):
    if SolverType == 'scipy':
        return True, scipy.sparse.linalg.spsolve(A, b)
    elif SolverType == 'QR':
        return True, QRsolve(np.array(A, dtype=complex), np.array(b, dtype=complex))
    elif SolverType == 'LU':
        LUres, P = LU(A)
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
        if abs(A[imax, i]) < tol:
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
        if abs(A[P[imax], i]) < tol:
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
        if abs(A[imax, i]) < tol:
            return None, None
        if imax != i:
            P[i], P[imax] = P[imax], P[i]
            A[i], A[imax] = A[imax].copy(), A[i].copy()
            P[N] += 1
        count = 0
        for j in range(i+1, N):
            if np.abs(A[j, i]) >= tol:
                A[j, i] /= A[i, i]
                for k in range(i+1, N):
                    if np.abs(A[j, k]) >= tol:
                        A[j, k] -= A[j, i] * A[i, k]
                        count +=1
    return A, P


"""
    This function implements the salme ILU(0) algorithm as ILU0_slow(A) but it was vectorized to be faster. 
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
        if np.abs(A[P[imax], i]) < tol:
            imax = i
        if imax != i:
            P[i], P[imax] = P[imax], P[i]
            P[N] += 1
        idx1 = np.nonzero(np.abs(A[P[i+1:N], i]) >= tol)[0]
        A[P[i + 1 + idx1], i:i+1] /= A[P[i], i]
        idx2 = np.nonzero(np.abs(A[P[i + 1 + idx1], i + 1:]) >= tol)
        A[P[i+1+idx2[0]], i+1+idx2[1]] -= np.outer(A[P[i+1:N], i], A[P[i], i+1:])[idx2]
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
            if np.abs(A[P[imax], i]) < tol:
                return None, None
        else: imax = i
        if imax != i:
            P[i], P[imax] = P[imax], P[i]
            P[N] += 1
        idx1 = np.nonzero(np.abs(A[P[i+1:N], i]) >= tol)[0]
        A[P[i + 1 + idx1], i:i+1] /= A[P[i], i]
        idx2 = np.nonzero(np.abs(A[P[i + 1 + idx1], i + 1:]) >= tol)
        A[P[i+1+idx2[0]], i+1+idx2[1]] -= np.outer(A[P[i+1:N], i], A[P[i], i+1:])[idx2]
    return A, P


"""
    This function implements the QR decomposition of A in a slow manner (only the third loop was vectorized)
"""
def QR_slow(A):
    M, N = np.shape(A)
    Q = np.zeros((M, N), dtype=complex)
    R = np.zeros((N, N), dtype=complex)
    for i in range(N):
        R[i, i] = np.linalg.norm(A[:, i], ord=2)
        Q[:, i] = A[:, i] / R[i, i]
        for j in range(i+1, N):
            R[i, j] = np.dot(Q[:, i].conjugate(), A[:, j])
            A[:, j] -= R[i, j]*Q[:, i]
    return Q, R


"""
    This function implements the same algorithm as the QR_slow(A) function but was fully vectorized
"""
def QR_columns(A):
    M, N = np.shape(A)
    # Q = np.zeros((M, N), dtype=complex)
    R = np.zeros((N, N), dtype=complex)
    for i in range(N):
        R[i, i] = np.linalg.norm(A[:, i], ord=2)
        A[:, i] = A[:, i] / R[i, i]
        R[i, i+1:] = np.dot(A[:, i].conjugate(), A[:, i+1:])
        A[:, i+1:] -= np.outer(A[:, i], R[i, i+1:])
    return A, R


"""
    This function implements the same algorithm as the two above functions but transposes A at 
    the beginning and Q at the end. 
    This way, we can use the fact that matrices are stored by rows (and not by columns) and the fact 
    that operations on rows are faster than operations on columns
"""
def QR(A):
    A = np.array(A, dtype=complex).T
    M, N = np.shape(A)
    Q = np.zeros((M, N), dtype=complex)
    R = np.zeros((N, N), dtype=complex)
    for i in range(N):
        R[i, i] = np.linalg.norm(A[i], ord=2)
        Q[i] = A[i] / R[i, i]
        R[i, i+1:] = np.dot(A[i+1:, :], Q[i].conjugate())
        A[i+1:, :] -= np.outer(R[i, i+1:], Q[i])
    return Q.T, R


"""
    This function implements a solver for the QR decompostion. It solves the system : 
        R x = Q^* b
    knowing that R is an upper triangular matrix.
"""
def QRsolve(A, b):
    Q, R = QR_columns(A)
    M, N = np.shape(Q)
    y = np.dot(Q.T.conjugate(), b)
    x = np.zeros(N, dtype=complex)
    for i in range(N - 1, -1, -1):
        x[i:i + 1] = (y[i] - np.dot(R[i, i:], x[i:])) / R[i, i]
    return x