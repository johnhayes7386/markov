import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

N = 100_000

a = 0.9
lambda_f = 1000
lambda_s = 10

component = rng.random(N) < a
u = rng.random(N)

T = np.where(
    component,
    -np.log(1-u)/lambda_f,
    -np.log(1-u)/lambda_s
)

print("Empirical mean:", np.mean(T))
print("Theoretical mean:", 0.0109)

print("Empirical P(T > 50 ms):", np.mean(T > 0.05))
print("Theoretical:", 0.060653)

t = np.linspace(0, 0.5, 1000)

f = (a*lambda_f*np.exp(-lambda_f*t)
     + (1-a)*lambda_s*np.exp(-lambda_s*t))

plt.hist(T, bins=200, density=True,
         alpha=0.5, label="Empirical")

plt.plot(t, f, label="Theory")

plt.yscale("log")
plt.xlabel("Dwell time (seconds)")
plt.ylabel("Density")
plt.legend()
plt.show()