import time

import numpy as np
from physics import *
from Solver import *
from util import format_time
import matplotlib.pyplot as plt
# demag tensor setup


class Simulation(object):
    def __init__(self, n : np.array,
                 dx : np.array,
                 mu0 : float,
                 gamma : float,
                 MSat : float,
                 A : float,
                 Dind : float,
                 alpha : float,
                 solver : str,
                 starting_dt=1e-15,
                 tol = 0.0001,
                 calculate_demagnetization_tensor=True,
                 demag_tensor_file="demag_tensor.npy",
                 output_filename="sp4_{}_adaptive_extra_physics"
                 ):

        self.n = n
        self.dx = dx
        self.mu0 = mu0
        self.gamma = gamma
        self.ms = MSat
        self.A = A
        self.alpha = alpha
        self.Dind = Dind
        self.BSat = MSat
        self.MSat = MSat
        self.h_zee = 0
        self.AnisU = np.array([0,0,0])
        self.demag_tensor_file = demag_tensor_file
        self.calculate_demagnetization_tensor = calculate_demagnetization_tensor
        self.starting_time = time.time()

        def dm_dt_wrapper(m, h_zee=0.0):
            return dm_dt(m, simulation=self, h_zee=h_zee)


        if solver == "ode23":
            self.solver = ODE23(starting_dt=starting_dt,dm_dt=dm_dt_wrapper, tol=tol)
        elif solver == "ode45":
            self.solver = ODE45(starting_dt=starting_dt,dm_dt=dm_dt_wrapper, tol=tol)
        elif solver == "euler":
            #self.solver =
            pass
        else:
            raise NotImplemented("choice of a solver right now is ode23, ode45 and euler")

        self.output_filename = output_filename.format(solver)
        self.effects = [DemagnetizationField(self), ExchangeField(self), DMI(self), MagnetoCrystallineAnisotropy(self)]

        self._calculate_demagnetization_tensor()

        # initialize magnetization that relaxes into s-state
        self.m = np.zeros(n + (3,))
        self.m[1:-1, :, :, 0] = 1.0
        self.m[(-1, 0), :, :, 1] = 1.0

    def setAnisotropyVector(self, AnisU : np.array) -> None:
        self.AnisU = np.array(AnisU)
        self.AnisU = self.AnisU / np.sqrt(self.AnisU.T @ self.AnisU)

    def setMagnetoCrystallineConstantsWholeGeometry(self, Ku1, Ku2):
        self.Ku1 = np.ones(self.n) * Ku1
        self.Ku2 = np.ones(self.n) * Ku2

    def add_effect(self):
        pass

    def remove_effect(self):
        pass

    def _calculate_demagnetization_tensor(self):
        # setup demag tensor
        if self.calculate_demagnetization_tensor:
            print("Calculating the demagnetization tensor")
            self.n_demag = np.zeros([2 * i - 1 for i in self.n] + [6])
            for i, t in enumerate(((newell_f, 0, 1, 2), (newell_g, 0, 1, 2), (newell_g, 0, 2, 1),
                                   (newell_f, 1, 2, 0), (newell_g, 1, 2, 0), (newell_f, 2, 0, 1))):
                set_n_demag(i, t[1:], t[0], self)

            print(self.n_demag.shape)
            np.save(self.demag_tensor_file, self.n_demag)
        else:
            print(f"loading the demagnetization tensor from {self.demag_tensor_file}")
            self.n_demag = np.load(self.demag_tensor_file)

        self.m_pad = np.zeros([2 * i - 1 for i in self.n] + [3])
        self.f_n_demag = np.fft.fftn(self.n_demag, axes=list(filter(lambda i: self.n[i] > 1, range(3))))

    # compute effective field
    def h_eff(self):
        h_eff = self.effects[0].calculate()
        for effect in self.effects[1:]:
            h_eff += effect.calculate()
        return h_eff

    def relax_for(self, relaxation_time):
        alpha = 1.00
        print(f"[{format_time(self.starting_time)}] starting relaxation")

        cum_time = 0.
        dreport = relaxation_time / 10.
        report = 0

        while cum_time <= relaxation_time:
            if cum_time >= report:
                report += dreport
                print(f"[{format_time(self.starting_time)}] time : {cum_time}, relaxation step {self.solver.dt}")
            cum_time += self.solver.dt
            self.m = self.solver.step(self.m, self.h_zee)

    def run_for(self, running_time):
        ts, mxs, mys, mzs = [], [], [], []
        # starting simulation
        with open(f'data/{self.output_filename}.dat', 'w') as f:
            cum_time = 0.
            dreport = running_time / 10.
            report = 0
            while cum_time <= running_time:
                mx, my, mz = tuple(map(lambda i: np.mean(self.m[:, :, :, i]), range(3)))
                f.write("%f %f %f %f %f\n" % ((cum_time, mx, my, mz, self.solver.dt)))
                if cum_time >= report:
                    print("-"*32)
                    print(f"[{format_time(self.starting_time)}] simulation time t= {cum_time}, current stepsize = {self.solver.dt}")
                    print(f"avg mx : {mx} | avg my : {my} | avg mz : {mz}")
                    print("-"*32)
                    report += dreport

                ts.append(cum_time)
                mxs.append(mx)
                mys.append(my)
                mzs.append(mz)
                cum_time += self.solver.dt

                # solver method
                self.m = self.solver.step(self.m,self.h_zee)

        plt.plot(ts, mxs, label="Mx", color="red")
        plt.plot(ts, mys, label="My", color="green")
        plt.plot(ts, mzs, label="Mz", color="blue")
        plt.legend()
        plt.savefig(f"images/M_{self.output_filename}.png")
        plt.show()