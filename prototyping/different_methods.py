import matplotlib.pyplot as plt 
import numpy as np
from solvers import *
from math import asinh, atan, sqrt, pi
from time import time

# f function (change of magnetiation with respect to time)
def dm_dt(m, h_zee = 0.):
    global h_eff, gamma, alpha
    h = h_eff(m) + h_zee
    dmdt = - gamma/(1+alpha**2) * np.cross(m, h) - alpha*gamma/(1+alpha**2) * np.cross(m, np.cross(m, h))
    return dmdt

# integrate using euler
def llg_euler(m, dt, h_zee=0.0):
    new_m = m +  dt * dm_dt(m, h_zee=h_zee)
    return new_m/np.repeat(np.sqrt((new_m*new_m).sum(axis=3)), 3).reshape(m.shape)
    
# integrate using midpoint
def llg_midpoint(m, dt, h_zee= 0.0):
    f_y_n = dm_dt(m, h_zee=h_zee)
    new_m = m + dt * dm_dt(m + 0.5 * dt * f_y_n, h_zee=h_zee)
    return new_m/np.repeat(np.sqrt((new_m*new_m).sum(axis=3)), 3).reshape(m.shape)

# integrate using Bogacki-Shampine method
def ode23(m, dt, h_zee=0.0):
  k1 = dm_dt(m, h_zee=h_zee)
  k2 = dm_dt(m + 0.5 * dt * k1, h_zee=h_zee)
  k3 = dm_dt(m + 0.75 * dt * k2, h_zee=h_zee)
  y__n_p_1 = m + 2./9. * dt * k1 + 1./3.*dt*k2 + 4./9. * dt * k3
 
  k4 = dm_dt(y__n_p_1, h_zee=h_zee)
  z__n_p_1 = m + 7./24.*dt*k1 + 0.25*dt*k2 + 1./3.*dt*k3 + 1./8.*dt*k4

  return z__n_p_1

# a very small number
eps = 1e-18

def format_time(starting_time):
    v = round(time() - starting_time, 3)
    return f"{v//3600}h {(v % 3600)//60}m {round(v % 60, 3)}s"

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
def set_n_demag(c, permute, func):
  it = np.nditer(n_demag[:,:,:,c], flags=['multi_index'], op_flags=['writeonly'])
  while not it.finished:
    value = 0.0
    for i in np.rollaxis(np.indices((2,)*6), 0, 7).reshape(64, 6):
      idx = list(map(lambda k: (it.multi_index[k] + n[k] - 1) % (2*n[k] - 1) - n[k] + 1, range(3)))
      value += (-1)**sum(i) * func(list(map(lambda j: (idx[j] + i[j] - i[j+3]) * dx[j], permute)))
    it[0] = - value / (4 * pi * np.prod(dx))
    it.iternext()

# compute effective field (demag + exchange)
def h_eff(m):
  # demag field
  m_pad[:n[0],:n[1],:n[2],:] = m
  f_m_pad = np.fft.fftn(m_pad, axes = list(filter(lambda i: n[i] > 1, range(3))))
  f_h_demag_pad = np.zeros(f_m_pad.shape, dtype=f_m_pad.dtype)
  f_h_demag_pad[:,:,:,0] = (f_n_demag[:,:,:,(0, 1, 2)]*f_m_pad).sum(axis = 3)
  f_h_demag_pad[:,:,:,1] = (f_n_demag[:,:,:,(1, 3, 4)]*f_m_pad).sum(axis = 3)
  f_h_demag_pad[:,:,:,2] = (f_n_demag[:,:,:,(2, 4, 5)]*f_m_pad).sum(axis = 3)

  h_demag = np.fft.ifftn(f_h_demag_pad, axes = list(filter(lambda i: n[i] > 1, range(3))))[:n[0],:n[1],:n[2],:].real

  # exchange field
  h_ex = - 2 * m * sum([1/x**2 for x in dx])
  for i in range(6):
    ## ???
    # h_ex = 100, 25, 1, 3
    if n[i % 3] == 1:
        repeats = 1
    else:
        repeats = [i//3*2] + [1]*(n[i%3]-2) + [2-i//3*2]
    v = np.repeat(m, repeats, axis = i%3) / dx[i%3]**2

    h_ex += v

  return ms*h_demag + 2*A/(mu0*ms)*h_ex

# setup mesh and material constants
n     = (100, 25, 1)
dx    = (5e-9, 5e-9, 3e-9)
mu0   = 4e-7 * pi
gamma = 2.211e5
ms    = 8e5
A     = 1.3e-11
alpha = 0.02
solver = ode23
output_filename = f"sp4_{solver.__name__}"

# setup demag tensor
n_demag = np.zeros([2*i-1 for i in n] + [6])
for i, t in enumerate(((f,0,1,2),(g,0,1,2),(g,0,2,1),(f,1,2,0),(g,1,2,0),(f,2,0,1))):
  set_n_demag(i, t[1:], t[0])

starting_time = time()

m_pad     = np.zeros([2*i-1 for i in n] + [3])
f_n_demag = np.fft.fftn(n_demag, axes = list(filter(lambda i: n[i] > 1, range(3))))

# initialize magnetization that relaxes into s-state
m = np.zeros(n + (3,))
m[1:-1,:,:,0]   = 1.0
m[(-1,0),:,:,1] = 1.0

# relax
alpha = 1.00
print(f"[{format_time(starting_time)}] starting relaxation")
for i in range(5000): 
  if i % 500 == 0:
    print(f"[{format_time(starting_time)}] relaxation step {i}")
  m = solver(m, 2e-13)

# switch
alpha = 0.02
dt    = 5e-15
h_zee = np.tile([-24.6e-3/mu0, +4.3e-3/mu0, 0.0], np.prod(n)).reshape(m.shape)

sim_time = 1e-9
steps = sim_time / dt

ts, mxs, mys, mzs = [], [], [], []
# starting simulation
with open(f'{output_filename}.dat', 'w') as f:
  for i in range(int(1e-9/dt)):
    if i % (steps // 10) == 0:
        print(f"[{format_time(starting_time)}] simulation time t= {i * dt}")

    mx, my, mz = tuple(map(lambda i: np.mean(m[:,:,:,i]), range(3)))
    f.write("%f %f %f %f\n" % ((i*1e9*dt,mx,my, mz)))

    ts.append(i)
    mxs.append(mx)
    mys.append(my)
    mzs.append(mz)

    # solver method
    m = solver(m, dt, h_zee)

plt.plot(ts, mys, label="My", color="green")
plt.legend()
plt.savefig(f"fMy_{output_filename}.png")
plt.show()