import numpy as np
from math import asinh, atan, sqrt, pi

v = 1
def set_n_demag(c, permute, func, n_demag, dx):
    global v
    it = np.nditer(n_demag[:, :, :, c], flags=['multi_index'], op_flags=['writeonly'])
    j = 0
    while not it.finished:
        #print(it.multi_index)
        value = 0.0

        for i in np.rollaxis(np.indices((2,) * 6), 0, 7).reshape(64, 6):
            idx = list(map(lambda k: (it.multi_index[k] + n[k] - 1) % (2 * n[k] - 1) - n[k] + 1, range(3)))

            value += (-1) ** sum(i) * func(list(map(lambda j: (idx[j] + i[j] - i[j + 3]) * dx[j], permute)))

        value=v
        #it[0] = - value / (4 * pi * np.prod(dx))
        it[0] = value
        it.iternext()
        j += 1
    v = v + 1

def identity(p):
    return 1

f = identity
g = identity

def calculate_demag_tensor(n, dx):
    print("Calculating the demagnetization tensor")
    n_demag = np.zeros([2 * i - 1 for i in n] + [6])

    for i, t in enumerate(((f, 0, 1, 2), (g, 0, 1, 2), (g, 0, 2, 1), (f, 1, 2, 0), (g, 1, 2, 0), (f, 2, 0, 1))):
        set_n_demag(i, t[1:], t[0], n_demag=n_demag, dx=dx)
    return n_demag


n     = (20, 10, 1)
dx    = (5e-9, 5e-9, 3e-9)
n_demag = calculate_demag_tensor(n ,dx)
print(n_demag)