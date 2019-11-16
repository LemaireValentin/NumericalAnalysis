from ndt import ndtfun
from mysolve import *
import time
import numpy as np
import matplotlib.pyplot as plt

# TESTS
A = np.array([[2.00,1.00,1.00,1.00],[4.00,3.00,3.00,1.00],[8.00,7.00,9.00,5.00],[6.00,7.00,9.00,8.00]], dtype=float)
b = np.array([1.00,2.00,3.00, 4.00])

num_nodes = []
times = []
for raff in np.arange(1, 1.8, 0.1):
    A, b, nodes, sol, cond = ndtfun(0.2, raff, 0, 0, 100.)
    print('NUM NODES ============', nodes)
    A, b = np.array(A), np.array(b)
    num_nodes.append(nodes)
    tic = time.time()
    mysolve(A, b)
    toc = time.time()
    times.append(toc - tic)

plt.title('Temps d\'exécution de LUsolve en fonction de la taille du maillage')
plt.ylabel('Temps [s]')
plt.xlabel('Nombre de noeuds [/]')
plt.plot(num_nodes, times, label='computed')
vec = np.linspace(num_nodes[0], num_nodes[-1], 200)
plt.plot(vec, np.polyval(np.polyfit(num_nodes, times, 3), vec), label='Expectation deg 3')
plt.plot(vec, np.polyval(np.polyfit(num_nodes, times, 2), vec), label='Expectation deg 2')
plt.legend()
plt.show()

logs = []
for a, b, c, d in zip(times[:-1], times[1:], num_nodes[:-1], num_nodes[1:]):
  logs.append(np.log(b/a)/np.log(d/c))

print(np.polyfit(num_nodes, times, 3))
print(np.polyfit(num_nodes, times, 2))
print(times)
print(num_nodes)
print(logs)
print(np.mean(logs))
