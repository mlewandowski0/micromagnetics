import numpy as np
from types import FunctionType
from math import sqrt

class Solver(object):
    def __init__(self, starting_dt : float, dm_dt : FunctionType ):
        self.dt = starting_dt
        self.dm_dt = dm_dt

    def step(self, m: np.array, h_zee : np.array) -> np.array:
        raise NotImplemented()


# integrate using Bogacki-Shampine method ( but adaptive step size)
class ODE23(Solver):
    def __init__(self, starting_dt : float, dm_dt : FunctionType, tol=0.0001):
        super().__init__(starting_dt=starting_dt, dm_dt=dm_dt)
        self.tol = 0.0001

    def step(self, m: np.array, h_zee : np.array) -> np.array:

        dt = self.dt

        k1 = self.dm_dt(m, h_zee=h_zee)
        k2 = self.dm_dt(m + 0.5 * dt * k1, h_zee=h_zee)
        k3 = self.dm_dt(m + 0.75 * dt * k2, h_zee=h_zee)
        y__n_p_1 = m + 2. / 9. * dt * k1 + 1. / 3. * dt * k2 + 4. / 9. * dt * k3

        k4 = self.dm_dt(y__n_p_1, h_zee=h_zee)
        z__n_p_1 = m + 7. / 24. * dt * k1 + 0.25 * dt * k2 + 1. / 3. * dt * k3 + 1. / 8. * dt * k4

        tau__n_p_1 = z__n_p_1 - y__n_p_1

        # Root Mean Squared Error
        avg_numerical_error = sqrt(np.sum(tau__n_p_1 ** 2) / np.prod(tau__n_p_1.shape))

        new_dt = 0.9 * dt * min(
            max(
                sqrt(self.tol / (2 * avg_numerical_error)),
                0.3
            ),
            2)

        self.dt = new_dt

        return z__n_p_1


# integrate using Dormand-Prince ( but adaptive step size)
class ODE45(Solver):
    def __init__(self, starting_dt : float, dm_dt : FunctionType, tol=0.0001):
        super().__init__(starting_dt=starting_dt, dm_dt=dm_dt)
        self.tol = 0.0001
        # coefficients for Dormand-Prince method
        self.k2_coeff = [1. / 5.]
        self.k3_coeff = [3. / 40., 9. / 40.]
        self.k4_coeff = [44. / 45., -56. / 15., 32. / 9.]
        self.k5_coeff = [19372. / 6561., -25360. / 2187., 64448. / 6561., -212. / 729.]
        self.k6_coeff = [9017. / 3168., -355. / 33., 46732. / 5247., 49. / 176., -5103. / 18656.]
        self.k7_coeff = [35. / 384., 0., 500. / 1113., 125. / 192., -2187 / 6784, 11 / 84]
        self.z__n_p_1_coeff = [5179. / 57600., 0. , 7571./195.  , 393./640.	, -92097./339200., 187./2100. , 1./40.]

    def step(self, m: np.array, h_zee) -> np.array:
        h = self.dt

        k1 = self.dm_dt(m, h_zee=h_zee)
        k2 = self.dm_dt(m + self.k2_coeff[0] * h * k1, h_zee=h_zee)
        k3 = self.dm_dt(m + self.k3_coeff[0] * h * k1 + self.k3_coeff[1] * h * k2, h_zee=h_zee)
        k4 = self.dm_dt(m + self.k4_coeff[0] * h * k1 + self.k4_coeff[1] * h * k2 + self.k4_coeff[2] * h * k3, h_zee=h_zee)

        k5 = self.dm_dt(m + self.k5_coeff[0] * h * k1 + self.k5_coeff[1] * h * k2 + self.k5_coeff[2] * h * k3
                   + self.k5_coeff[3] * h * k4, h_zee=h_zee)

        k6 = self.dm_dt(m + self.k6_coeff[0] * h * k1 + self.k6_coeff[1] * h * k2 + self.k6_coeff[2] * h * k3 +
                   self.k6_coeff[3] * h * k4 + self.k6_coeff[4] * h * k5, h_zee=h_zee)

        y__n_p_1 = m + self.k7_coeff[0] * h * k1 + self.k7_coeff[1] * h * k2 + self.k7_coeff[2] * h * k3 + \
                   self.k7_coeff[3] * h * k4 + self.k7_coeff[4] * h * k5 + self.k7_coeff[5] * h * k6

        k7 = self.dm_dt(y__n_p_1, h_zee=h_zee)

        z__n_p_1 = m + self.z__n_p_1_coeff[0] * k1 * h + self.z__n_p_1_coeff[1] * k2 * h + self.z__n_p_1_coeff[2] * k3 * h + \
                   self.z__n_p_1_coeff[3] * k4 * h + self.z__n_p_1_coeff[4] * k5 * h + self.z__n_p_1_coeff[5] * k6 * h + \
                   self.z__n_p_1_coeff[6] * k7 * h

        tau__n_p_1 = z__n_p_1 - y__n_p_1

        # Root Mean Squared Error
        avg_numerical_error = sqrt(np.sum(tau__n_p_1 ** 2) / np.prod(tau__n_p_1.shape))

        # print(avg_numerical_error)

        self.dt = 0.9 * self.dt * min(
            max(
                sqrt(self.tol / (2 * avg_numerical_error)),
                0.3
            ),
            2)

        return z__n_p_1