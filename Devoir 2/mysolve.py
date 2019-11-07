
# The function mysolve(A, b) is invoked by ndt.py
# to solve the linear system
# Implement your solver in this file and then run:
# python ndt.py


SolverType = 'scipy'

import scipy.sparse
import scipy.sparse.linalg

def mysolve(A, b):
    if SolverType == 'scipy':
        return True, scipy.sparse.linalg.spsolve(A, b)
    #elif SolverType == 'QR':
       # write here your code for the QR solver
    elif SolverType == 'LU':
        return False, 0
        # write here your code for the LU solver
    #elif SolverType == 'GMRES':
        # write here your code for the LU solver
    else:
        return False, 0
        
