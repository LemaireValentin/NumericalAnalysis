import numpy as np
import time
from mysolve import *
from ndt import ndtfun
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy


def ILU0_slow(A):
    N = len(A)
    for i in range(N):
        if abs(A[i, i]) == 0:
            return None, None
        for j in range(i+1, N):
            if np.abs(A[j, i]) > 0:
                A[j, i] /= A[i, i]
                for k in range(i+1, N):
                    if np.abs(A[j, k]) > 0:
                        A[j, k] -= A[j, i] * A[i, k]
    return A


M = np.array([[1, 4, 0, 0, 0, 0, -3, 0, 2, 0],
              [1, 1, 0, 0, 5, 0, 1, 0, 0, 10],
              [0, 0, 1, 0, -31, 0, 4, 0, 0, 0],
              [0, 0, 0, 20, 3, 1, 0, 0, 7, 0],
              [0, 2, 1, -3, 2, 8, 0, 0, 0, 6],
              [0, 0, 0, 12, 2, 4, 0, 0, 0, 0],
              [4, 2, 5, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 4, 6, 9],
              [1, 0, 0, 2, 0, 0, 0, 4, 60, 0],
              [0, 3, 0, 0, 3, 0, 0, 8, 0, 14]])

def test_solve():
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, 1, 50, 1, 100., run=True, copy=True, SolverType='GMRES', rtol=1e-17, prec=True)
    print("Precision = ", np.linalg.norm(A @ sol - b)/np.linalg.norm(b))



def plot_eig():
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, 1, 50, 0, 100., run=False, copy=True, SolverType='GMRES', rtol=1e-7, prec=True)
    A2 = A.copy()
    LUres= ILU0_slow(np.array(A))
    U = np.triu(LUres)
    L = np.eye(len(LUres), dtype=complex) + LUres - U
    M = L.conjugate() @ U
    Minv = np.linalg.inv(M)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 7))

    s = np.linalg.eigvals(A2)

    ticks_x = ticker.FuncFormatter(lambda x, pos: '{:5.1e}'.format(x))
    ax1.xaxis.set_major_formatter(ticks_x)

    ticks_y = ticker.FuncFormatter(lambda x, pos: '{:5.1e}'.format(x))
    ax1.yaxis.set_major_formatter(ticks_y)

    ax1.scatter(s.real, s.imag)
    ax1.set_xlabel("Axe Réel")
    ax1.set_ylabel("Axe Imaginaire")
    ax1.set_title("Valeurs propres de A")
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    s2 = np.linalg.eigvals(Minv @ A2)
    ax2.scatter(s2.real, s2.imag)
    ax2.set_xlabel("Axe Réel")
    ax2.set_ylabel("Axe Imaginaire")
    ax2.set_title("Valeurs propres de M^(-1) A")
    ax2.set_yscale('log')
    ax2.set_xscale('log')

    plt.show()


def get_conv(ref, max_iter, rtol):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    prec = False
    word = 'sans'

    # STATIQUE
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 0, 0, 100., run=False, copy=True, SolverType='numpy', rtol=1e-7, prec=prec)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), rtol, prec, max_iter=max_iter)
    ax1.plot(np.arange(len(res)), res/np.linalg.norm(b), label='Statique')
    print(len(res))

    # STATIONNAIRE
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 0, 1, 100., run=False, copy=True, SolverType='numpy', rtol=1e-7, prec=prec)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), rtol, prec, max_iter=max_iter)
    ax1.plot(np.arange(len(res)), res/np.linalg.norm(b), label='Stationnaire')

    # HARMONIQUE
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 50, 0, 100., run=False, copy=True, SolverType='numpy', rtol=1e-7, prec=prec)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), rtol, prec, max_iter=max_iter)
    ax1.plot(np.arange(len(res)), res/np.linalg.norm(b), label='Harmonique')

    # DYNAMIQUE
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 50, 1, 100., run=False, copy=True, SolverType='numpy', rtol=1e-7, prec=prec)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), rtol, prec, max_iter=max_iter)
    ax1.plot(np.arange(len(res)), res/np.linalg.norm(b), label='Dynamique')

    ax1.set_yscale('log')
    ax1.set_title("Convergence de GMRES "+word+" préconditionnement")
    ax1.set_xlabel("Nombre d'itérations [/]")
    ax1.set_ylabel("||r||/||b||  [/]")
    ax1.legend()
    ax1.grid()

    prec = True
    word='avec'

    # STATIQUE
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 0, 0, 100., run=False, copy=True, SolverType='numpy', rtol=1e-7, prec=prec)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), rtol, prec, max_iter=max_iter)
    ax2.plot(np.arange(len(res)), res/np.linalg.norm(b), label='Statique')

    # STATIONNAIRE
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 0, 1, 100., run=False, copy=True, SolverType='numpy', rtol=1e-7, prec=prec)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), rtol, prec, max_iter=max_iter)
    ax2.plot(np.arange(len(res)), res/np.linalg.norm(b), label='Stationnaire')

    # HARMONIQUE
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 50, 0, 100., run=False, copy=True, SolverType='numpy', rtol=1e-7, prec=prec)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), rtol, prec, max_iter=max_iter)
    ax2.plot(np.arange(len(res)), res/np.linalg.norm(b), label='Harmonique')

    # DYNAMIQUE
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 50, 1, 100., run=False, copy=True, SolverType='numpy', rtol=1e-7, prec=prec)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), rtol, prec, max_iter=max_iter)
    ax2.plot(np.arange(len(res)), res/np.linalg.norm(b), label='Dynamique')

    ax2.set_yscale('log')
    ax2.set_title("Convergence de GMRES "+word+" préconditionnement")
    ax2.set_xlabel("Nombre d'itérations [/]")
    ax2.set_ylabel("||r||/||b||  [/]")
    ax2.legend()
    ax2.grid()


    plt.show()


def plot_prec_iter():
    prec = []
    iter = []
    for n in range(1, 41, 1):
        A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, 1, 50, 1, 100., run=False, copy=True, SolverType='numpy',
                                                    rtol=1e-7, prec=False)
        b = b.copy()
        sA, iA, jA = CSRformat(np.array(A))
        u, res = csrGMRES(sA, iA, jA, np.array(b), rtol=1e-14, prec=True, max_iter=n)
        prec.append(np.linalg.norm(np.dot(A, u) - b)/np.linalg.norm(b))
        iter.append(len(res))

    prec = np.array(prec)
    iter = np.array(iter)
    idx = np.where(np.logical_and(prec >= 1e-9, prec <= 1e-1))
    plt.plot(iter[idx], prec[idx], label="Préconditionné")
    prec = []
    iter = []
    for n in range(1, 242, 6):
        A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, 1, 50, 1, 100., run=False, copy=True, SolverType='numpy',
                                                    rtol=1e-7, prec=False)
        b = b.copy()
        sA, iA, jA = CSRformat(np.array(A))
        u, res = csrGMRES(sA, iA, jA, np.array(b), rtol=1e-7, prec=False, max_iter=n)
        prec.append(np.linalg.norm(np.dot(A, u) - b)/np.linalg.norm(b))
        iter.append(len(res))

    prec = np.array(prec)
    iter = np.array(iter)
    idx = np.where(np.logical_and(prec >= 1e-9, prec <= 1e-1 ))
    plt.plot(iter[idx], prec[idx], label="Sans préconditionnement")

    plt.title("Précision relative de la solution en fonction du nombre d'itérations")
    plt.ylabel("Précision relative [/]")
    plt.xlabel("Nombre d'itérations [/]")
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.show()

def get_iter_prec(rtol, prec, ref, max_iter):
    precision = []
    niter = []

    # STATIQUE
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 0, 0, 100., run=False, copy=True, SolverType='numpy',
                                                rtol=1e-7, prec=prec)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), rtol, prec, max_iter=max_iter)
    precision.append(np.linalg.norm(np.dot(A, u) - b)/np.linalg.norm(b))
    niter.append(len(res))

    # STATIONNAIRE
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 0, 1, 100., run=False, copy=True, SolverType='numpy',
                                                rtol=1e-7, prec=prec)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), rtol, prec, max_iter=max_iter)
    precision.append(np.linalg.norm(np.dot(A, u) - b)/np.linalg.norm(b))
    niter.append(len(res))

    # HARMONIQUE
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 50, 0, 100., run=False, copy=True, SolverType='numpy',
                                                rtol=1e-7, prec=prec)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), rtol, prec, max_iter=max_iter)
    precision.append(np.linalg.norm(np.dot(A, u) - b)/np.linalg.norm(b))
    niter.append(len(res))

    # DYNAMIQUE
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 50, 1, 100., run=False, copy=True, SolverType='numpy',
                                                rtol=1e-7, prec=prec)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), rtol, prec, max_iter=max_iter)
    precision.append(np.linalg.norm(np.dot(A, u) - b)/np.linalg.norm(b))
    niter.append(len(res))

    print(precision)
    print(niter)

def plot_prec(ref):
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 50, 1, 100., run=False, copy=True, SolverType='numpy',
                                                rtol=1e-20, prec=False)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), 1e-20, True)
    print(res)
    plt.plot(np.arange(len(res)), res, label='Préconditionné')
    A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, ref, 50, 1, 100., run=False, copy=True, SolverType='numpy',
                                                rtol=1e-17, prec=False)
    sA, iA, jA = CSRformat(A)
    u, res = csrGMRES(sA, iA, jA, np.array(b), 1e-17, False)
    plt.plot(np.arange(len(res)), res, label='Non-préconditionné')

    plt.title("Précision relative de la solution en fonction du nombre d'itérations")
    plt.yscale('log')
    plt.xlabel("Nombre d'itérations [/]")
    plt.ylabel("Précision relative [/]")
    plt.legend()
    plt.grid()
    plt.show()

# get_conv(ref=1, max_iter=300, rtol=1e-11)
test_solve()
# plot_prec_iter()
# plot_eig()
# get_iter_prec(1e-7, False, 1, 300)
# plot_prec(ref=1)
