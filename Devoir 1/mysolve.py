
# The function mysolve(A, b) is invoked by ndt.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ndt.py


SolverType = 'scipy'

import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import sys


def mysolve(A, b):

    if SolverType == 'scipy':
        return True, scipy.sparse.linalg.spsolve(A, b),
    #elif SolverType == 'QR':
       # write here your code for the QR solver
    elif SolverType == 'LU':
        return False, 0
        # write here your code for the LU solver
    #elif SolverType == 'GMRES':
        # write here your code for the LU solver
    else:
        return False, 0


def isSparse(A):
    count = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i, j].imag == 0 and A[i, j].real == 0:
                count+=1
    return count/(len(A)*len(A[0])) >= 0.3

def isComplex(A):
    return np.any(A.imag > 0)


def isReal(A):
    return np.all(A.imag == 0)


def isSym(A):
    return np.all(1e-17 >= abs((A - A.T).imag)) and np.all(1e-17 >= abs((A - A.T).real))


def isUnit(m):
    return np.allclose(np.eye(len(m)), m.dot(m.H).real, rtol=1e-17) and np.allclose(np.zeros(len(m)), m.dot(m.T.conj()).imag, rtol=1e-17)

def isHermite(A):
    return np.all(1e-17 >= abs((A-A.H).imag)) and np.all(1e-17 >= abs((A-A.H).real))


def isInversible(A):
    return np.linalg.cond(A) < 1 / sys.float_info.epsilon


def isDefPos(A):
    if isReal(svd(A)):
        return np.all(svd(A) > 0)
    return False

def isBands(A):
    for l in range(min(len(A), len(A[0]))):
        bln = True
        for i in range(len(A)):
            for j in range(0, max(0, i-l)):
                if abs(A[i, j].real) >= 1e-17 or abs(A[i, len(A[0])-j].real) >= 1e-17 or abs(A[i, j].imag) >= 1e-17 or abs(A[i, len(A[0])-j].imag) >= 1e-17:
                    bln = False
        if bln:
            return True, l
    return False, -1



def svd(A):
    return scipy.linalg.svd(A, compute_uv=False)


