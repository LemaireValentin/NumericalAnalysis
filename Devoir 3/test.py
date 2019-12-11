from ndt import ndtfun
from mysolve import *
import time
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap


def csr_format_test():
  A = np.array([[10, 20, 0, 0, 0, 0], [0, 30, 0, 40, 0, 0], [0, 0, 50, 60, 70, 0], [0, 0, 0, 0, 0, 80]])
  print(CSRformat(A))


def test_csr_bands():
  A = np.array([[1, 2, 0], [0, 3, 0], [0, 0, 4]])
  sA, iA, jA = CSRformat(A)
  # print(bands_csr(iA, jA))
  A = np.array([[8, 3, 0, 0, 0], [1, 9, 4, 0, 7], [0, 5, 6, 2, 0], [0, 0, 0, 4, 0], [0, 4, 2, 0, 1]], dtype=complex)
  print(A.real)
  sA, iA, jA = CSRformat(A)
  print(LUcsr(sA, iA, jA))
  LUres, P = LU(A)
  print(print(LUres[P[:len(A)]].real))


def test_solve():
  A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, 1, 50, 0, 100., True, False, 'numpy')


def test_rcmk():
  M = np.array([[0, 4, 0, 0, 0, 0, -3, 0, 2, 0],
                [1, 0, 0, 0, 5, 0, 1, 0, 0, 10],
                [0, 0, 0, 0, -31, 0, 4, 0, 0, 0],
                [0, 0, 0, 0, 3, 1, 0, 0, 7, 0],
                [0, 2, 1, -3, 0, 8, 0, 0, 0, 6],
                [0, 0, 0, 12, 2, 0, 0, 0, 0, 0],
                [4, 2, 5, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 6, 9],
                [1, 0, 0, 2, 0, 0, 0, 4, 0, 0],
                [0, 3, 0, 0, 3, 0, 0, 8, 0, 0]])
  b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  sA, iA, jA = CSRformat_slow(M)
  #print(sA, iA, jA)
  r = RCMK(iA, jA)
  #print(r)

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
  b = np.ones(len(M))

  print(mysolve(M, b, 'LUcsr-rcmk-slow'))

  # r_inv = invert_r(r)
  # print(M[np.ix_(r, r)])
  # sA, iA, jA = reduce_bands(sA, iA, jA, r, r_inv)
  # print(sA, iA, jA)
  # sLU, iLU, jLU = LUcsr_opt(sA, iA, jA)
  # print(LUsolve_csr(sLU, iLU, jLU, b[r])[r_inv])
  # print(np.linalg.solve(M, b))


def showmat(A):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.set_aspect('equal')
  plt.imshow(abs(A) <= 1e-15, interpolation='nearest', cmap='gray')
  plt.show()


def test_rcmk_matrix():
  A, b, num_nodes, sol, cond, tictoc = ndtfun(0.2, 1, 0, 0, 100, False, True, 'LUcsr-rcmk')
  A2 = A.copy()
  sA, iA, jA = CSRformat(A)
  R = RCMK(iA, jA)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
  ax1.spy(A, precision=1e-15)
  ax1.set_ylabel('Lignes de la matrice')
  ax1.set_xlabel('Colonnes de la matrice')
  LUres = LU_no_pivot(A)
  ax2.spy(LUres, precision=1e-15)
  ax2.set_ylabel('Lignes de la matrice')
  ax2.set_xlabel('Colonnes de la matrice')
  plt.show()

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
  ax1.spy(A2[np.ix_(R,R)], precision=1e-15)
  ax1.set_ylabel('Lignes de la matrice')
  ax1.set_xlabel('Colonnes de la matrice')
  LUres = LU_no_pivot(A2[np.ix_(R, R)])
  ax2.spy(LUres, precision=1e-15)
  ax2.set_ylabel('Lignes de la matrice')
  ax2.set_xlabel('Colonnes de la matrice')
  plt.show()



def test_complexity_maillage(start, target, step, num):
  num_nodes = []
  times_LUcsr_rcmk = []
  for raff in np.arange(start, target, step):
    times_loop = []
    for i in range(num):
      A, b, nodes, sol, cond, tictoc = ndtfun(0.2, raff, 50, 0, 100., False, False, 'LUcsr-rcmk')
      times_loop.append(tictoc)
    num_nodes.append(nodes)
    times_LUcsr_rcmk.append(np.mean(times_loop))
  print("Times LUcsr RCMK = ", times_LUcsr_rcmk)

  times_LUcsr = []
  for raff in np.arange(start, target, step):
    times_loop = []
    for i in range(num):
      A, b, nodes, sol, cond, tictoc = ndtfun(0.2, raff, 50, 0, 100., False, False, 'LUcsr')
      times_loop.append(tictoc)
    times_LUcsr.append(np.mean(times_loop))
  print("Times LUcsr = ", times_LUcsr)

  times_LU = []
  for raff in np.arange(start, target, step):
    times_loop = []
    for i in range(num):
      A, b, nodes, sol, cond, tictoc = ndtfun(0.2, raff, 50, 0, 100., False, False, 'LU-no-pivot')
      times_loop.append(tictoc)
    times_LU.append(np.mean(times_loop))
  print("Times LU = ", times_LU)

  plt.title("Temps d\'exécution des solveurs (lents)\n%s" % "\n".join(
    wrap(" en fonction de la taille du maillage", width=60)))
  plt.ylabel('Temps [s]')
  plt.xlabel('Nombre de noeuds [/]')
  print("num nodes ", num_nodes)
  print("times LUcsr rcmk", times_LUcsr_rcmk)
  print("times LUcsr", times_LUcsr)
  print("times LU", times_LU)
  print("rapport", np.array(times_LUcsr)/np.array(times_LU))
  plt.plot(num_nodes, times_LUcsr_rcmk, label='LUcsr avec RCMK')
  plt.plot(num_nodes, times_LUcsr, label='LUcsr sans RCMK')
  plt.plot(num_nodes, times_LU, label='LU plein')

  plt.yscale('log')
  plt.xscale('log')
  plt.grid()
  plt.legend()
  plt.show()


def test_precision_and_time(ref):
    precision_LU = []
    precision_np = []
    times = []
    order = ['statique', 'stationnaire', 'harmonique', 'dynamique']

    SolverType = 'LUcsr-rcmk'

    A, b, nodes, x_lu, cond, tictoc = ndtfun(0.2, ref, 0, 0, 100., False, True, SolverType)
    A2 = A.copy()
    times.append(tictoc)
    precision_LU.append(np.linalg.norm(np.dot(A2, x_lu) - b) / np.linalg.norm(b))
    x_np = np.linalg.solve(A, b)
    precision_np.append(np.linalg.norm(np.dot(A, x_np) - b) / np.linalg.norm(b))

    A, b, nodes, x_lu, cond, tictoc = ndtfun(0.2, ref, 0, 100, 100., False, True, SolverType)
    A2 = A.copy()
    times.append(tictoc)
    precision_LU.append(np.linalg.norm(np.dot(A2, x_lu) - b) / np.linalg.norm(b))
    x_np = np.linalg.solve(A, b)
    precision_np.append(np.linalg.norm(np.dot(A, x_np) - b) / np.linalg.norm(b))

    A, b, nodes, x_lu, cond, tictoc = ndtfun(0.2, ref, 50, 0, 100., False, True, SolverType)
    A2 = A.copy()
    times.append(tictoc)
    precision_LU.append(np.linalg.norm(np.dot(A2, x_lu) - b) / np.linalg.norm(b))
    x_np = np.linalg.solve(A, b)
    precision_np.append(np.linalg.norm(np.dot(A, x_np) - b) / np.linalg.norm(b))

    A, b, nodes, x_lu, cond, tictoc = ndtfun(0.2, ref, 30, 100, 100., False, True, SolverType)
    A2 = A.copy()
    times.append(tictoc)
    precision_LU.append(np.linalg.norm(np.dot(A2, x_lu) - b) / np.linalg.norm(b))
    x_np = np.linalg.solve(A, b)
    precision_np.append(np.linalg.norm(np.dot(A, x_np) - b) / np.linalg.norm(b))

    print(order)
    print(precision_LU)
    # print(precision_np)
    print(times)
    print(nodes)


def test_perf_convert(start, target, step):
  num_nodes = []
  times_slow = []
  times_fast = []
  for raff in np.arange(start, target, step):
    A, b, nodes, sol, cond, tictoc = ndtfun(0.2, raff, 50, 0, 100., False, True, 'numpy')
    num_nodes.append(nodes)
    tic = time.time()
    sA, iA, jA = CSRformat_slow(A)
    toc = time.time()
    times_slow.append(toc-tic)
    tic = time.time()
    sA, iA, jA = CSRformat(A)
    toc = time.time()
    times_fast.append(toc-tic)

  print(num_nodes)
  print(times_slow)
  print(times_fast)

  fig, ax1 = plt.subplots()
  line, = ax1.plot(num_nodes, times_slow, color='b')
  ax1.set_yscale('log')
  ax1.set_ylabel('temps [s]', color='b')
  ax1.set_xscale('log')
  ax1.set_xlabel('Nombre de noeuds [/]')

  ax2 = ax1.twinx()
  line2, = ax2.plot(num_nodes, times_fast, color='r')
  ax2.set_yscale('log')
  ax2.set_ylabel('temps [s]', color='r')
  ax2.set_xscale('log')


  plt.legend((line, line2), ('Convertisseur lent', 'Convertisseur rapide'))
  plt.title('Performances des convertisseurs en format creux')
  plt.grid()
  plt.show()


def show_convert_graphs():
  num_nodes = np.array([143, 273, 393, 512, 610, 853, 1033, 1250, 1486, 1748, 2008, 2529, 2887, 3196, 3595, 4046, 4367, 5020, 5524])
  times_slow = np.array([0.0445561408996582, 0.13568902015686035, 0.25207090377807617, 0.42189908027648926, 0.6056649684906006, 1.2791249752044678, 1.7608609199523926, 2.8911690711975098, 4.364299058914185, 5.56797194480896, 7.510974884033203, 11.317915201187134, 14.974966764450073, 17.767994165420532, 22.04378390312195, 28.012226104736328, 32.19191288948059, 47.946707010269165, 58.274182081222534])
  #times_fast = np.array([0.0011403560638427734, 0.005254030227661133, 0.004119873046875, 0.0060999393463134766, 0.00843501091003418, 0.014618158340454102, 0.0191190242767334, 0.03339815139770508, 0.03657817840576172, 0.048142194747924805, 0.06706476211547852, 0.09434700012207031, 0.12487292289733887, 0.14425992965698242, 0.18429803848266602, 0.2277359962463379, 0.25883007049560547, 0.3659329414367676, 0.4095029830932617])

  # times_slow = np.array([0.035993099212646484, 0.12372899055480957, 0.25958871841430664, 0.4226682186126709, 0.62471604347229, 1.1814792156219482, 1.767928123474121, 3.555328845977783, 4.091328859329224, 5.405625820159912, 6.775190114974976, 10.725812911987305, 14.104809999465942, 17.129799127578735, 29.642412900924683, 28.216601848602295, 32.70568799972534, 46.173280000686646, 55.058631896972656])
  times_fast = np.array([0.0011987686157226562, 0.0027358531951904297, 0.00403904914855957, 0.006075859069824219, 0.008014917373657227, 0.013741016387939453, 0.018773794174194336, 0.029107093811035156, 0.0398099422454834, 0.04747891426086426, 0.06159615516662598, 0.09297013282775879, 0.11869978904724121, 0.14285588264465332, 0.1816408634185791, 0.22325992584228516, 0.2605471611022949, 0.35030102729797363, 0.41749000549316406])

  fig, ax1 = plt.subplots()
  line, = ax1.plot(num_nodes, times_slow, 'c--')
  ax1.plot()
  ax1.set_yscale('log')
  ax1.set_xscale('log')
  ax1.set_xlabel('Nombre de noeuds [/]')

  vec = np.linspace(num_nodes[0], num_nodes[-1], 200)
  log_line = lambda x, m: np.exp(m[0] * np.log(x) + m[1])
  m = np.polyfit(np.log(num_nodes), np.log(times_slow), 1)
  # ax1.plot(vec, log_line(vec, m), 'c--')

  # ax2 = ax1.twinx()
  line2, = ax1.plot(num_nodes, times_fast, 'g--')
  ax1.set_yscale('log')
  ax1.set_ylabel('temps [s]')
  ax1.set_xscale('log')
  m = np.polyfit(np.log(num_nodes), np.log(times_fast), 1)
  # ax1.plot(vec, log_line(vec, m), 'g--')

  vec2 = np.linspace(num_nodes[0], num_nodes[0] + 1000, 200)
  ax1.plot(vec2, log_line(vec2, [2, m[1]]))

  plt.legend((line, line2), ('Convertisseur lent', 'Convertisseur rapide'))
  plt.title('Performances des convertisseurs en format creux')
  plt.grid()
  plt.show()

def test_max_neighbors(start, target, step):
  neighbors = []
  num_nodes = []
  for raff in np.arange(start, target, step):
    A, b, nodes, sol, cond, tictoc = ndtfun(0.2, raff, 50, 0, 100., False, True, 'numpy')
    # num_nodes.append(nodes)
    neighbors_it = [np.count_nonzero(A[i]) for i in range(len(A))]
    num_nodes.append(nodes)
    neighbors.append(np.max(neighbors_it))
  print(neighbors)
  print(num_nodes)

def show_two_mats():
  A, b, nodes, sol, cond, tictoc = ndtfun(0.2, 2, 50, 0, 100., False, True, 'numpy')
  A = np.array(A)
  plt.spy(A)
  plt.title("Masque de A avant permutation par RCMK")
  plt.show()
  sA, iA, jA = CSRformat(A)
  R = RCMK(iA, jA)
  plt.spy(A[np.ix_(R, R)])
  plt.title("Masque de A après permutation par RCMK")
  plt.show()


def show_graphs():
  # --------- SLOW ---------
  num_nodes = np.array([143, 273, 393, 512, 610, 853, 1033, 1250])
  times_LUcsr_rcmk = np.array([0.1107170581817627, 0.2649233341217041, 0.6455850601196289, 0.9196529388427734, 1.2176897525787354, 4.371031045913696, 7.062726974487305, 5.562952995300293])
  times_LUcsr = np.array([0.8127319812774658, 5.108612298965454, 16.394694089889526, 34.821744203567505, 61.767162799835205, 187.7172920703888, 311.3622291088104, 544.3353657722473])
  times_LU =  np.array([0.4147980213165283, 2.8725268840789795, 10.532163858413696, 22.596881866455078, 36.80120897293091, 105.51872706413269, 185.54832792282104, 338.2360701560974])

  # --------- FAST ---------
  num_nodes = np.array([143, 273, 393, 512, 610, 853, 1033, 1250])
  times_LUcsr_rcmk = np.array([0.016343116760253906, 0.03119206428527832, 0.06372809410095215, 0.09501218795776367, 0.10574197769165039, 0.2554659843444824, 0.3693382740020752, 0.4061911106109619])
  times_LUcsr = np.array([0.023453950881958008, 0.12381291389465332, 0.32711291313171387, 0.7912440299987793, 1.3185851573944092, 3.593208074569702, 6.7734668254852295, 13.51212191581726])
  times_LU=np.array([0.00882101058959961, 0.040882110595703125, 0.14716315269470215, 0.20949387550354004, 0.4222750663757324, 1.5444610118865967, 3.0385591983795166, 5.06761908531189])


  fig, ax = plt.subplots()

  ax.set_title("Temps d\'exécution des solveurs (optimisés)\n%s" % "\n".join(
    wrap(" en fonction de la taille du maillage", width=60)))
  ax.set_ylabel('Temps [s]')
  ax.set_xlabel('Nombre de noeuds [/]')
  ax.plot(num_nodes, times_LUcsr_rcmk,'c--', label='LUcsr avec RCMK')
  ax.plot(num_nodes, times_LUcsr, 'g--', label='LUcsr sans RCMK')
  ax.plot(num_nodes, times_LU, 'y--', label='LU plein')


  vec = np.linspace(num_nodes[0], num_nodes[-1], 200)
  log_line = lambda x, m: np.exp(m[0] * np.log(x) + m[1])

  m = np.polyfit(np.log(num_nodes), np.log(times_LUcsr_rcmk), 1)
  # ax.plot(vec, log_line(vec, m), 'c--')
  vec2 = np.linspace(num_nodes[0], num_nodes[0] + 1000, 200)


  m = np.polyfit(np.log(num_nodes), np.log(times_LUcsr), 1)
  # ax.plot(vec, log_line(vec, m), 'g--')

  m = np.polyfit(np.log(num_nodes), np.log(times_LU), 1)
  # ax.plot(vec, log_line(vec, m), 'y--')
  # ax.plot(vec2, log_line(vec2, [3, m[1]]))

  ax.set_yscale('log')
  ax.set_xscale('log')
  # ax.set_aspect('equal')

  # plt.xlim(10, 10000)
  plt.grid()
  plt.legend()
  plt.show()


def test_sparsity(start, stop, step):
  num_nodes = []
  band_before_rcmk = []
  band_after_rcmk = []
  percent_zeros_before = []
  percent_zeros_no_rcmk = []
  percent_zeros_with_rcmk = []


  for raff in np.arange(start, stop, step):
    A, b, nodes, sol, cond, tictoc = ndtfun(0.2, raff, 50, 0, 100., False, True, 'numpy')
    num_nodes.append(nodes)
    A = np.array(A)
    A2 = np.copy(A)
    sA, iA, jA = CSRformat(np.array(A))
    r = RCMK(iA, jA)
    r_inv = invert_r(r)
    sAr, iAr, jAr = reduce_bands(sA, iA, jA, r, r_inv)
    band_before_rcmk.append(max(compute_bands(iA, jA, len(A))))
    band_after_rcmk.append(max(compute_bands(iAr, jAr, len(A))))
    percent_zeros_before.append((np.count_nonzero(np.array(A))/(len(A)*len(A)))*100)
    LUres = LU_no_pivot(A)
    percent_zeros_no_rcmk.append((np.count_nonzero(np.array(LUres))/(len(A)*len(A)))*100)
    LUres_rcmk = LU_no_pivot(A2[np.ix_(r, r)])
    percent_zeros_with_rcmk.append((np.count_nonzero(LUres_rcmk)/(len(A)*len(A)))*100)
    if raff == 1:
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
      # fig.suptitle("Génération du fill-in avec et sans RCMK")
      ax1.spy(LUres)
      ax2.spy(LUres_rcmk)

      plt.show()

  print(num_nodes)
  print(band_before_rcmk)
  print(band_after_rcmk)
  print(percent_zeros_before)
  print(percent_zeros_no_rcmk)
  print(percent_zeros_with_rcmk)

def test_precision(ref):
  precision = []
  A, b, nodes, sol, cond, tictoc = ndtfun(0.2, ref, 50, 0, 100., False, True, 'numpy')
  A = np.array(A)
  b = np.array(b)
  precision.append(np.linalg.norm(A@sol - b)/np.linalg.norm(b))

  A, b, nodes, sol, cond, tictoc = ndtfun(0.2, ref, 50, 0, 100., False, True, 'LU')
  A = np.array(A)
  b = np.array(b)
  precision.append(np.linalg.norm(A@sol - b)/np.linalg.norm(b))

  A, b, nodes, sol, cond, tictoc = ndtfun(0.2, ref, 50, 0, 100., False, True, 'LUcsr')
  A = np.array(A)
  b = np.array(b)
  precision.append(np.linalg.norm(A@sol - b)/np.linalg.norm(b))

  A, b, nodes, sol, cond, tictoc = ndtfun(0.2, ref, 50, 0, 100., False, True, 'LUcsr-rcmk')
  A = np.array(A)
  b = np.array(b)
  precision.append(np.linalg.norm(A@sol - b)/np.linalg.norm(b))

  print(precision)

# test_rcmk()
# test_csr_bands()
# test_solve()
# test_rcmk_matrix()
# test_complexity_maillage(0.5, 2.26, 0.25, 1)
test_precision_and_time(2)
# test_max_neighbors(0.5, 5, 0.2)
# show_graphs()
# test_perf_convert(0.5, 5.1, 0.25)
# show_convert_graphs()
# show_two_mats()
# test_sparsity(0.5, 2.1, 0.5)
# test_precision(2)



