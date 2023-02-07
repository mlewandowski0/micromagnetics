import matplotlib.pyplot as plt

with open("sp4_euler.dat") as f:
    Mx, My, Mz, t = [], [], [], []
    for line in f:
        v = line.strip().split(" ")
        t.append(float(v[0]))
        Mx.append(float(v[1]))
        My.append(float(v[2]))
        Mz.append(float(v[3]))

plt.plot(t, My, label="My", color="green")
plt.legend()
plt.savefig("sp4_euler.png")
plt.show()

plt.plot(t, Mx, label="Mx", color="red")
plt.plot(t, My, label="My", color="green")
plt.plot(t, Mz, label="Mz", color="blue")
plt.legend()
plt.show()