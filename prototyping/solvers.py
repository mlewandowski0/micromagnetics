import numpy as np

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
