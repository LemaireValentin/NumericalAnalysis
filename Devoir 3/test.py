import numpy as np
from ndt import ndtfun
from mysolve import *

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
  print(LUcsr_opt(sA, iA, jA))
  LUres, P = LU(A)
  print(print(LUres[P[:len(A)]].real))


test_csr_bands()