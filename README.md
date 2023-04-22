# micromagnetics
numerical methods for LLG ( micromagnetics framework )

## This folder contains the computational part of my 3rd year project
This publication https://arxiv.org/abs/1411.7188 contains very basic, working with python2 code for micromagnetics. I deided to improve upon it to learn more about micromagnetics : <Br>
the whole changelist :
- Calculation of the demagnetization tensor was ported to GPU (very expensive calculation done before on CPU, taking a lot of time)
- solver was changed from Euler to adaptive timestep ( Dormand-Prince / Bogacki-Shampine) and higher order methods like (RK4)
- more physics terms were added : Magnetocrystalline anistropy and DMI
- various optimizations of code, like using JIT ( Just-In-Time) compliation using numba and porting timestepping to run on GPU

# Folders :
prototyping/ : 
- contains basically all the code developed ( all the different features are labelled). For example paper_code.py is the starting code. adaptive_solver.py is a an improvement from paper_code.py, where I use the adaptive timestep method (BS or DOPRI). Final code with all features is in "adaptive_solver_extra_physics_demag_gpu.py"
mumax3/ :
- contains some mumax3 standard problem3 simulations, and interaction with mumax3 through python
refactored :
- making the outputs from prototyping look much better and be reusable ( in OOP way)
