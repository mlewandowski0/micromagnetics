import matplotlib.pyplot as plt

v1 = open("data/sp4_ode23_adaptive.dat")
v2 = open("data/sp4_ode23_adaptive_extra_physics.dat")

v1_data = v1.readlines()
v2_data = v2.readlines()
ts = []
diffs = []

for i in range(len(v1_data)):
    fls1 = [float(v) for v in v1_data[i].split()]
    fls2 = [float(v) for v in v2_data[i].split()]

    ts.append(fls1[0])
    print(fls1)

v1.close()
v2.close()