import fidimag
from fidimag.common import CuboidMesh
from fidimag.micro import Sim, UniformExchange, Demag, DMI, UniaxialAnisotropy
mesh = CuboidMesh(nx=60, ny=60, nz=1, dx=2.0, dy=2.0, dz=2.0, unit_length=1e-9)

def Ms_init(position):
    """
    Set where the system has magnetic material
    Form a nanodisk shape
    """
    Ms = 8.6e5
    x, y, z = position
    if (x - 60)**2 + (y - 60)**2 < 60**2:
        return Ms
    else:
        return 0

def m_init(position):
    """
    Approximate skyrmion profile
    """
    x, y, z = position
    if (x - 60)**2 + (y - 60)**2 < 40**2:
        return (0, 0, 1)
    else:
        return (0, 0, -1)

sim = Sim(mesh, name='target_skyrmion')
sim.set_Ms(Ms_init)
sim.set_m(m_init)
sim.add(Demag())
sim.add(UniformExchange(A=1e-11))
sim.add(DMI(D=3e-3))
sim.add(UniaxialAnisotropy(Ku=4e5, axis=(0, 0, 1)))
sim.relax()
sim.save_vtk()
