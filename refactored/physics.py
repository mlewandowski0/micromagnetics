import numpy as np
from math import atan, asinh, sqrt, pi

# a very small number
EPS = 1e-18

# newell f
def newell_f(p: np.array):
    # print(type(p))
    x, y, z = abs(p[0]), abs(p[1]), abs(p[2])
    return + y / 2.0 * (z ** 2 - x ** 2) * asinh(y / (sqrt(x ** 2 + z ** 2) + EPS)) \
        + z / 2.0 * (y ** 2 - x ** 2) * asinh(z / (sqrt(x ** 2 + y ** 2) + EPS)) \
        - x * y * z * atan(y * z / (x * sqrt(x ** 2 + y ** 2 + z ** 2) + EPS)) \
        + 1.0 / 6.0 * (2 * x ** 2 - y ** 2 - z ** 2) * sqrt(x ** 2 + y ** 2 + z ** 2)


# newell g
def newell_g(p: np.array):
    x, y, z = p[0], p[1], abs(p[2])
    return + x * y * z * asinh(z / (sqrt(x ** 2 + y ** 2) + EPS)) \
        + y / 6.0 * (3.0 * z ** 2 - y ** 2) * asinh(x / (sqrt(y ** 2 + z ** 2) + EPS)) \
        + x / 6.0 * (3.0 * z ** 2 - x ** 2) * asinh(y / (sqrt(x ** 2 + z ** 2) + EPS)) \
        - z ** 3 / 6.0 * atan(x * y / (z * sqrt(x ** 2 + y ** 2 + z ** 2) + EPS)) \
        - z * y ** 2 / 2.0 * atan(x * z / (y * sqrt(x ** 2 + y ** 2 + z ** 2) + EPS)) \
        - z * x ** 2 / 2.0 * atan(y * z / (x * sqrt(x ** 2 + y ** 2 + z ** 2) + EPS)) \
        - x * y * sqrt(x ** 2 + y ** 2 + z ** 2) / 3.0

def set_n_demag(c, permute, func, simulation):
    it = np.nditer(simulation.n_demag[:, :, :, c], flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        value = 0.0
        for i in np.rollaxis(np.indices((2,) * 6), 0, 7).reshape(64, 6):
            idx = list(map(lambda k: (it.multi_index[k] + simulation.n[k] - 1) % (2 * simulation.n[k] - 1) - simulation.n[k] + 1, range(3)))
            value += (-1) ** sum(i) * func(list(map(lambda j: (idx[j] + i[j] - i[j + 3]) * simulation.dx[j], permute)))
        it[0] = - value / (4 * pi * np.prod(simulation.dx))
        it.iternext()


# f function (change of magnetisation with respect to time)
def dm_dt(m, simulation ,  h_zee=0.):
    # global h_eff, gamma, alpha
    h = simulation.h_eff() + h_zee
    dmdt = - simulation.gamma / (1 + simulation.alpha ** 2) * np.cross(m, h) - simulation.alpha * simulation.gamma \
           / (1 + simulation.alpha ** 2) * np.cross(m, np.cross(m, h))
    return dmdt

class Effect(object):
    def __init__(self, simulation):
        self.simulation = simulation

    def calculate(self):
        raise NotImplemented


class DemagnetizationField(Effect):
    def calculate(self):
        # demag field
        self.simulation.m_pad[:self.simulation.n[0], :self.simulation.n[1], :self.simulation.n[2], :] = self.simulation.m
        f_m_pad = np.fft.fftn(self.simulation.m_pad, axes=list(filter(lambda i: self.simulation.n[i] > 1, range(3))))
        f_h_demag_pad = np.zeros(f_m_pad.shape, dtype=f_m_pad.dtype)
        f_h_demag_pad[:, :, :, 0] = (self.simulation.f_n_demag[:, :, :, (0, 1, 2)] * f_m_pad).sum(axis=3)
        f_h_demag_pad[:, :, :, 1] = (self.simulation.f_n_demag[:, :, :, (1, 3, 4)] * f_m_pad).sum(axis=3)
        f_h_demag_pad[:, :, :, 2] = (self.simulation.f_n_demag[:, :, :, (2, 4, 5)] * f_m_pad).sum(axis=3)

        h_demag = np.fft.ifftn(f_h_demag_pad, axes=list(filter(lambda i: self.simulation.n[i] > 1, range(3))))[:self.simulation.n[0], :self.simulation.n[1], :self.simulation.n[2],:].real
        B_demag = self.simulation.ms * h_demag

        return B_demag


class ExchangeField(Effect):
    def calculate(self):
        # exchange field
        h_ex = - 2 * self.simulation.m * sum([1 / x ** 2 for x in self.simulation.dx])
        for i in range(6):
            ## ???
            # h_ex = 100, 25, 1, 3
            if self.simulation.n[i % 3] == 1:
                repeats = 1
            else:
                repeats = [i // 3 * 2] + [1] * (self.simulation.n[i % 3] - 2) + [2 - i // 3 * 2]
            v = np.repeat(self.simulation.m, repeats, axis=i % 3) / self.simulation.dx[i % 3] ** 2
            h_ex += v
        B_exch = 2 * self.simulation.A / (self.simulation.mu0 * self.simulation.ms) * h_ex
        return B_exch

class DMI(Effect):
    def calculate(self):
        B_dm = np.zeros_like(self.simulation.m)
        mx = self.simulation.m[:, :, :, 0]
        my = self.simulation.m[:, :, :, 1]
        mz = self.simulation.m[:, :, :, 2]
        delta_x, delta_y, delta_z = self.simulation.dx[0], self.simulation.dx[1], self.simulation.dx[2]

        # calculate derivatives using central difference
        dmz_dx = np.zeros_like(mz)
        dmz_dx[1:-1, :, :] = (mz[2:, :, :] - mz[:-2, :, :]) / (2 * delta_x)
        # apply boundary conditions
        #                 mz here ? or mx
        dmz_dx[0, :, :] = mz[0, :, :] + self.simulation.Dind / (2 * self.simulation.A) * mx[0, :, :] * delta_x
        dmz_dx[-1, :, :] = mz[-1, :, :] + self.simulation.Dind / (2 * self.simulation.A) * mx[-1, :, :] * delta_x

        dmz_dy = np.zeros_like(mz)
        dmz_dy[:, 1:-1, :] = (mz[:, 2:, :] - mz[:, :-2, :]) / (2 * delta_y)
        # apply boundary conditions
        #                 mz here ? or my
        dmz_dy[:, 0, :] = mz[:, 0, :] + self.simulation.Dind / (2 * self.simulation.A) * my[:, 0, :] * delta_y
        dmz_dy[:, -1, :] = mz[:, -1, :] + self.simulation.Dind / (2 * self.simulation.A) * my[:, -1, :] * delta_y

        dmx_dx = np.zeros_like(mx)
        dmx_dx[1:-1, :, :] = (mx[2:, :, :] - mx[:-2, :, :]) / (2 * delta_x)
        # apply boundary conditions
        dmx_dx[0, :, :] = mx[0, :, :] - self.simulation.Dind / (2 * self.simulation.A) * mz[0, :, :] * delta_x
        dmx_dx[-1, :, :] = mx[-1, :, :] - self.simulation.Dind / (2 * self.simulation.A) * mz[-1, :, :] * delta_x

        dmy_dy = np.zeros_like(my)
        dmy_dy[:, 1:-1, :] = (my[:, 2:, :] - my[:, :-2, :]) / (2 * delta_y)
        # apply boundary conditions
        dmy_dy[:, 0, :] = my[:, 0, :] - self.simulation.Dind / (2 * self.simulation.A) * mz[:, 0, :] * delta_y
        dmy_dy[:, -1, :] = my[:, -1, :] - self.simulation.Dind / (2 * self.simulation.A) * mz[:, -1, :] * delta_y

        B_dm[:, :, :, 0] = dmz_dx
        B_dm[:, :, :, 1] = dmz_dy
        B_dm[:, :, :, 2] = -dmx_dx - dmy_dy
        B_dm = 2 * self.simulation.Dind / self.simulation.ms * B_dm
        return B_dm

class MagnetoCrystallineAnisotropy(Effect):
    def calculate(self):
        # Magneto-crystalline anisotropy
        u_dot_m = np.sum(self.simulation.m * self.simulation.AnisU.reshape(1, 1, 1, 3), axis=-1)
        magnetocrystalline_1st_term = 2 * self.simulation.Ku1 / self.simulation.BSat * u_dot_m
        shape = magnetocrystalline_1st_term.shape
        magnetocrystalline_1st_term = magnetocrystalline_1st_term.reshape(shape[0], shape[1], shape[2], 1)
        magnetocrystalline_1st_term = magnetocrystalline_1st_term * self.simulation.AnisU.reshape(1, 1, 1, 3)

        magnetocrystalline_2nd_term = 4 * self.simulation.Ku2 / self.simulation.BSat * np.power(u_dot_m, 3)
        shape = magnetocrystalline_2nd_term.shape
        magnetocrystalline_2nd_term = magnetocrystalline_2nd_term.reshape(shape[0], shape[1], shape[2], 1)
        magnetocrystalline_2nd_term = magnetocrystalline_2nd_term * self.simulation.AnisU.reshape(1, 1, 1, 3)

        B_anis = magnetocrystalline_1st_term + magnetocrystalline_2nd_term
        return B_anis