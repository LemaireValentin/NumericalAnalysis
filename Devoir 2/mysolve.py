import scipy.sparse
import scipy.sparse.linalg
import numpy as np

# The function mysolve(A, b) is invoked by ndt.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ndt.py

SolverType = 'LU'


def mysolve(A, b):
    if SolverType == 'scipy':
        return True, scipy.sparse.linalg.spsolve(A, b)
    elif SolverType == 'QR':
        return True, QRsolve(A, b)
    elif SolverType == 'LU':
        LUres, P = ILU0_slow(A)
        #if not LUres:
        #    return False, 0
        return True, LUsolve(LUres, b, P)
    elif SolverType == 'GMRES':
        return False, 0
    else:
        return False, 0


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


def LUsolve(A, b, P):
    N = len(A)
    y = np.zeros(N)
    for i in range(N):
        y[i:i+1] = (b[P[i]] - np.dot(A[P[i], :i+1], y[:i+1]))
    x = np.zeros(N)
    for i in range(N-1, -1, -1):
        x[i:i+1] = (y[i] - np.dot(A[P[i], i:], x[i:])) / A[P[i], i]
    return x


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
        print(count)
    return A, P


def ILU0(A):
    tol = 1e-15
    N = len(A)
    P = np.arange(N + 1)
    P[N] = 0
    for i in range(N):
        imax = i + np.argmax(np.abs(A[P[i:N], i]))
        if np.abs(A[P[imax], i]) < tol:
            return None, None
        if imax != i:
            P[i], P[imax] = P[imax], P[i]
            P[N] += 1
        idx1 = np.nonzero(np.abs(A[P[i+1:N], i]) >= tol)[0]
        A[P[i + 1 + idx1], i:i+1] /= A[P[i], i]
        idx2 = np.nonzero(np.abs(A[P[i + 1 + idx1], i + 1:]) >= tol)
        A[P[i+1+idx2[0]], i+1+idx[1]] -= np.outer(A[P[i+1:N], i], A[P[i], i+1:])[idx2]
    return A, P


def QRfactorize(A):
    M, N = np.shape(A)
    Q = np.zeros((M, N))
    R = np.zeros((N, N))
    for i in range(N):
        R[:i, i] = np.einsum('ji, j->i', Q[:, :i], A[:, i])
        v = A[:, i] - np.einsum('ij, j->i', Q[:, :i], R[:i, i])
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]
    return Q, R


def QR_slow(A):
    M, N = np.shape(A)
    Q = np.zeros((M, N))
    R = np.zeros((N, N))
    for i in range(N):
        Q[:, i] = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            Q[:, i] -= Q[:, j] * R[j, i]
        R[i, i] = np.linalg.norm(Q[:, i])
        Q[:, i] /= R[i, i]
    return Q, R

def QRsolve(A, b):
    Q, R = QRfactorize(A)
    M, N = np.shape(Q)
    y1 = np.einsum('ji, j->i', A, b)
    y2 = np.zeros(N)
    for i in range(N):
        y2[i] = (y1[i] - np.dot(R[:i, i], y2[:i])) / R[i, i]
    x = np.zeros(N)
    for i in range(N-1, -1, -1):
        x[i] = (y2[i] - np.dot(R[i, i:], x[i:])) / R[i, i]
    return x


# TESTS
test_mat = np.array([[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]], dtype=float)
A = np.array([[2.00,1.00,1.00,0.00],[4.00,3.00,3.00,1.00],[8.00,7.00,9.00,5.00],[6.00,7.00,9.00,8.00]], dtype=float)
test_mat_2 = np.copy(test_mat)
A_2 = np.copy(A)

A3 = np.array([[1, 0], [2, 1]], dtype=float)

A_qr = np.array([[-1, -1, 1], [1, 3, 3], [-1, -1, 5], [1, 3, 7]], dtype=float)
# print(QRsolve(A3, [3, 4]))

# print(LU(A))
# print(scipy.linalg.lu(A)[1], scipy.linalg.lu(A)[2])
# LU, P = LU(A3)
# print(LUsolve(A3, [3, 4], P))
# print(ILU0_slow(A_2))

