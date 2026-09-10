import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

N = 10_000
R = 1_000


C = np.ones(R, dtype=int)

for n in range(3, N):

   
    U = rng.random(R)

  
    grow = U < (C / n)


    C += grow

z = C / N

mean_z = np.mean(z)
ratio = np.std(z) / mean_z

print("Empirical mean z:", mean_z)
print("Theory mean:", 1/3)

print("Empirical SD/mean:", ratio)
print("Theory SD/mean:", 1/np.sqrt(2))

print("Smallest core:", np.min(C))
print("Largest core:", np.max(C))

# theoretical density
x = np.linspace(0, 1, 500)
h = 2*(1-x)

plt.hist(z, bins=30, density=True,
         alpha=0.5, label="Simulation")

plt.plot(x, h, label="h(z) = 2(1-z)")

plt.xlabel("z = C/N")
plt.ylabel("Density")
plt.legend()
plt.show()