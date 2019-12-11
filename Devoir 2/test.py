from ndt import ndtfun
from mysolve import *
import time
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap

# TESTS
A = np.array([[2.00,1.00,1.00,1.00],[4.00,3.00,3.00,1.00],[8.00,7.00,9.00,5.00],[6.00,7.00,9.00,8.00]], dtype=float)
b = np.array([1.00,2.00,3.00, 4.00])


def test_complexity_maillage(start, target, step, num):
    num_nodes = []
    times_QR = []
    times_LU = []
    for raff in np.arange(start, target, step):
        times_loop = []
        for i in range(num):
            A, b, nodes, sol, cond, tictoc = ndtfun(0.2, raff, 0, 0, 100., False, True, 'QR')
            times_loop.append(tictoc)
        num_nodes.append(nodes)
        times_QR.append(np.mean(times_loop))
    for raff in np.arange(start, target, step):
        times_loop = []
        for i in range(num):
            A, b, nodes, sol, cond, tictoc = ndtfun(0.2, raff, 0, 0, 100., False, True, 'LU')
            times_loop.append(tictoc)
        # num_nodes.append(nodes)
        times_LU.append(np.mean(times_loop))

    plt.title("Temps d\'exécution de LUsolve et QRsolve (optimisés)\n%s" % "\n".join(wrap(" en fonction de la taille du maillage", width=60)))
    plt.ylabel('Temps [s]')
    plt.xlabel('Nombre de noeuds [/]')
    plt.plot(num_nodes, times_QR, label='QRsolve')
    plt.plot(num_nodes, times_LU, label='LUsolve')

    vec = np.linspace(num_nodes[0], num_nodes[-1], 200)
    plt.plot(vec, np.polyval(np.polyfit(num_nodes, times_LU, 3), vec), label='Polyfit LU degré 3')
    # plt.plot(vec, np.polyval(np.polyfit(num_nodes, times_LU, 2), vec), label='Polyfit LU degré 2')
    # plt.plot(vec, np.polyval(np.polyfit(num_nodes, times_QR, 3), vec), label='Polyfit QR degré 3')
    # plt.plot(vec, np.polyval(np.polyfit(num_nodes, times_QR, 2), vec), label='Polyfit QR degré 2')

    plt.grid()
    plt.legend()
    plt.show()

    print(np.linalg.norm(np.polyval(np.polyfit(num_nodes, times_QR, 3), num_nodes) - times_QR))
    print(np.linalg.norm(np.polyval(np.polyfit(num_nodes, times_QR, 2), num_nodes) - times_QR))

    print(np.linalg.norm(np.polyval(np.polyfit(num_nodes, times_LU, 3), num_nodes) - times_LU))
    print(np.linalg.norm(np.polyval(np.polyfit(num_nodes, times_LU, 2), num_nodes) - times_LU))
    """
    logs = []
    for a, b, c, d in zip(times[:-1], times[1:], num_nodes[:-1], num_nodes[1:]):
      logs.append(np.log(b/a)/np.log(d/c))

    print(np.polyfit(num_nodes, times, 3))
    print(np.polyfit(num_nodes, times, 2))
    print(times)
    print(num_nodes)
    print(logs)
    print(np.mean(logs))
    """

def test_precision_and_time(ref):
    precision_LU = []
    precision_np = []
    times = []
    order = ['statique', 'stationnaire', 'harmonique', 'dynamique']

    SolverType = 'QR'

    A, b, nodes, x_lu, cond, tictoc = ndtfun(0.2, ref, 0, 0, 100., False, True, SolverType)
    A2 = A.copy()
    times.append(tictoc)
    precision_LU.append(np.linalg.norm(np.dot(A2, x_lu) - b)/np.linalg.norm(b))
    x_np = np.linalg.solve(A, b)
    precision_np.append(np.linalg.norm(np.dot(A, x_np) - b) / np.linalg.norm(b))

    A, b, nodes, x_lu, cond, tictoc = ndtfun(0.2, ref, 0, 100, 100., False, True, SolverType)
    A2 = A.copy()
    times.append(tictoc)
    precision_LU.append(np.linalg.norm(np.dot(A2, x_lu) - b)/np.linalg.norm(b))
    x_np = np.linalg.solve(A, b)
    precision_np.append(np.linalg.norm(np.dot(A, x_np) - b) / np.linalg.norm(b))

    A, b, nodes, x_lu, cond, tictoc = ndtfun(0.2, ref, 50, 0, 100., False, True, SolverType)
    A2 = A.copy()
    times.append(tictoc)
    precision_LU.append(np.linalg.norm(np.dot(A2, x_lu) - b)/np.linalg.norm(b))
    x_np = np.linalg.solve(A, b)
    precision_np.append(np.linalg.norm(np.dot(A, x_np) - b) / np.linalg.norm(b))

    A, b, nodes, x_lu, cond, tictoc = ndtfun(0.2, ref, 30, 100, 100., False, True, SolverType)
    A2 = A.copy()
    times.append(tictoc)
    precision_LU.append(np.linalg.norm(np.dot(A2, x_lu) - b)/np.linalg.norm(b))
    x_np = np.linalg.solve(A, b)
    precision_np.append(np.linalg.norm(np.dot(A, x_np) - b) / np.linalg.norm(b))

    print(order)
    print(precision_LU)
    print(precision_np)
    print(times)
    print(nodes)


def complexity_test():
    num_nodes = []
    times = []

    for n in np.arange(100, 500, 100):
        a = np.random.rand(n, n)
        b = np.random.rand(n)
        tic = time.time()
        mysolve(a, b)
        toc = time.time()
        num_nodes.append(n)
        times.append(toc-tic)
        print(n)

    # print(times[3]/times[1])
    # print(times[7]/times[3])
    plt.title('Temps d\'exécution de LUsolve en fonction de la taille du maillage')
    plt.ylabel('Temps [s]')
    plt.xlabel('Nombre de noeuds [/]')
    plt.plot(num_nodes, times, label='computed')
    vec = np.linspace(num_nodes[0], num_nodes[-1], 200)
    plt.plot(vec, np.polyval(np.polyfit(num_nodes, times, 3), vec), label='Expectation deg 3')
    plt.plot(vec, np.polyval(np.polyfit(num_nodes, times, 2), vec), label='Expectation deg 2')
    plt.legend()
    plt.show()
    print(times)
    print(np.linalg.norm(np.polyval(np.polyfit(num_nodes, times, 3), num_nodes) - times))
    print(np.linalg.norm(np.polyval(np.polyfit(num_nodes, times, 2), num_nodes) - times))


def complexity_test_200_400(num):
    tic1 = time.time()
    for _ in np.arange(num):
        a = np.random.rand(200, 200)
        b = np.random.rand(200)
        mysolve(a, b, 'LU')
    toc1 = time.time()
    time200 = toc1 - tic1

    tic2 = time.time()
    for _ in np.arange(num):
        a = np.random.rand(400, 400)
        b = np.random.rand(400)
        mysolve(a, b, 'LU')
    toc2 = time.time()
    time400 = toc2 - tic2

    print(time400 / time200)


def condMinvA(freq, vel):
    A, b, nodes, x_lu, cond, tictoc = ndtfun(0.2, 1, freq, vel, 100., False, True)
    A2 = A.copy()
    LUres, P = ILU0(np.array(A))
    LUres = LUres[P[:-1]]
    U = np.triu(LUres)
    L = np.eye(len(LUres), dtype=complex) + LUres - U
    M = L.conjugate() @ U
    Minv = np.linalg.inv(M)
    return np.linalg.cond(Minv.conjugate() @ A2), cond

def cond():
    cond = []
    condA = []
    order = ['statique', 'stationnaire', 'harmonique', 'dynamique']
    res = condMinvA(0, 0)
    cond.append(res[0])
    condA.append(res[1])

    res = condMinvA(0, 100)
    cond.append(res[0])
    condA.append(res[1])

    res = condMinvA(50, 0)
    cond.append(res[0])
    condA.append(res[1])

    res = condMinvA(50, 100)
    cond.append(res[0])
    condA.append(res[1])
    print(order)
    print(cond)
    print(condA)


# cond()
# complexity_test_200_400(10)
# test_complexity_maillage(0.3, 2.5, 0.2, 1)
# complexity_test()
test_precision_and_time(2)
# print(ndtfun(0.2, 2, 0, 0, 100, False, True, 'LU'))


