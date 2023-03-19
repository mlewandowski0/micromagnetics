# references : 
# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/93JB00694
# https://arxiv.org/pdf/1411.7188.pdf
# 
#
# fidimag : compute_demag_tensors_2dpbc

import numpy as np
from math import asinh, atan, sqrt, pi

# a very small number
eps = 1e-18
n     = (100, 25, 1)
dx    = (5e-9, 5e-9, 3e-9)
demag_tensor_file = "demag_tensor.npy"

# newell f
def f(p):
  #print(type(p))
  x, y, z = abs(p[0]), abs(p[1]), abs(p[2])
  return + y / 2.0 * (z**2 - x**2) * asinh(y / (sqrt(x**2 + z**2) + eps)) \
         + z / 2.0 * (y**2 - x**2) * asinh(z / (sqrt(x**2 + y**2) + eps)) \
         - x*y*z * atan(y*z / (x * sqrt(x**2 + y**2 + z**2) + eps))       \
         + 1.0 / 6.0 * (2*x**2 - y**2 - z**2) * sqrt(x**2 + y**2 + z**2)

# newell g
def g(p):
  x, y, z = p[0], p[1], abs(p[2])
  return + x*y*z * asinh(z / (sqrt(x**2 + y**2) + eps))                         \
         + y / 6.0 * (3.0 * z**2 - y**2) * asinh(x / (sqrt(y**2 + z**2) + eps)) \
         + x / 6.0 * (3.0 * z**2 - x**2) * asinh(y / (sqrt(x**2 + z**2) + eps)) \
         - z**3 / 6.0 * atan(x*y / (z * sqrt(x**2 + y**2 + z**2) + eps))        \
         - z * y**2 / 2.0 * atan(x*z / (y * sqrt(x**2 + y**2 + z**2) + eps))    \
         - z * x**2 / 2.0 * atan(y*z / (x * sqrt(x**2 + y**2 + z**2) + eps))    \
         - x*y * sqrt(x**2 + y**2 + z**2) / 3.0


# demag tensor setup
def set_n_demag(c, permute, func, n_demag, dx):
  it = np.nditer(n_demag[:,:,:,c], flags=['multi_index'], op_flags=['writeonly'])
  while not it.finished:
    value = 0.0
    for i in np.rollaxis(np.indices((2,)*6), 0, 7).reshape(64, 6):
      idx = list(map(lambda k: (it.multi_index[k] + n[k] - 1) % (2*n[k] - 1) - n[k] + 1, range(3)))
      value += (-1)**sum(i) * func(list(map(lambda j: (idx[j] + i[j] - i[j+3]) * dx[j], permute)))
    it[0] = - value / (4 * pi * np.prod(dx))
    it.iternext()

# demag tensor setup
def set_n_demag_test(c, permute, func, n_demag, dx):
  it = np.nditer(n_demag[:,:,:,c], flags=['multi_index'], op_flags=['writeonly'])
  while not it.finished:
    value = 0.0
    for i in np.rollaxis(np.indices((2,)*6), 0, 7).reshape(64, 6):
      idx = list(map(lambda k: (it.multi_index[k] + n[k] - 1) % (2*n[k] - 1) - n[k] + 1, range(3)))
      value += (-1)**sum(i) * func(list(map(lambda j: (idx[j] + i[j] - i[j+3]) * dx[j], permute)))
    it[0] = - value / (4 * pi * np.prod(dx))
    it.iternext()


def calculate_demag_tensor(n, dx):
    print("Calculating the demagnetization tensor")
    n_demag = np.zeros([2*i-1 for i in n] + [6])
    for i, t in enumerate(((f,0,1,2),(g,0,1,2),(g,0,2,1),(f,1,2,0),(g,1,2,0),(f,2,0,1))):
        set_n_demag(i, t[1:], t[0], n_demag=n_demag, dx=dx)

    print(n_demag.shape)
    np.save(demag_tensor_file, n_demag)


def calculate_demag_tensor_test(n, dx):
    print("Calculating the demagnetization tensor")
    n_demag = np.zeros([2*i-1 for i in n] + [6])
    for i, t in enumerate(((f,0,1,2),(g,0,1,2),(g,0,2,1),(f,1,2,0),(g,1,2,0),(f,2,0,1))):
        print(i, t)
        set_n_demag_test(i, t[1:], t[0], n_demag=n_demag, dx=dx)

    print(n_demag.shape)
    np.save(demag_tensor_file, n_demag)
    return n_demag