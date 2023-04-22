# standard problems https://www.ctcms.nist.gov/~rdm/mumag.org.html

import matplotlib.pyplot as plt
import numpy as np
from math import pi
from util import format_time

from Simulation import Simulation


# setup mesh and material constants
sim = Simulation(n=(100, 25, 1),
                 dx=(5e-9, 5e-9, 3e-9),
                 mu0 = 4e-7 * pi,
                 gamma=2.211e5,
                 MSat=800e3,
                 A=1.3e-11,
                 alpha=0.02,
                 Dind=3e-3,
                 solver="ode23",
                 calculate_demagnetization_tensor=False
                 )

sim.setAnisotropyVector([0, 0, 1])
sim.setMagnetoCrystallineConstantsWholeGeometry(Ku1=717e3,Ku2=0)

relaxation_time = 1e-9
simulation_time = 1e-9

# relax
sim.relax_for(relaxation_time)

# switch
sim.alpha = 0.02
sim.h_zee = np.tile([-24.6e-3/sim.mu0, +4.3e-3/sim.mu0, 0.0], np.prod(sim.n)).reshape(sim.m.shape)
sim_time = 1e-9
sim.run_for(sim_time)