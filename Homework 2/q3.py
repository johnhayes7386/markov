import numpy as np
import matplotlib.pyplot as plt
import time

def rejection_gamma(N, lam, seed=42):
    rng = np.random.default_rng(seed)

    c = 1 / (np.e * lam * (1-lam))

    samples = []
    proposals = 0

    start = time.perf_counter()

    while len(samples) < N:
        u1 = rng.random()
        x = -np.log(1-u1) / lam

        u2 = rng.random()

        f = x * np.exp(-x)
        g = lam * np.exp(-lam*x)

        proposals += 1

        if u2 < f/(c*g):
            samples.append(x)

    elapsed = time.perf_counter() - start

    samples = np.array(samples)

    acceptance = N / proposals
    time_per_sample = elapsed / N

    return samples, acceptance, time_per_sample, c


N = 10_000

for lam in [0.5, 0.2]:

    samples, acc, t, c = rejection_gamma(N, lam)

    print("lambda =", lam)
    print("Empirical acceptance =", acc)
    print("Theoretical acceptance =", 1/c)
    print("Mean time per accepted sample =", t)

    x = np.linspace(0, 12, 500)

    plt.hist(samples, bins=50, density=True,
             alpha=0.5, label="Empirical PDF")

    plt.plot(x, x*np.exp(-x),
             label="Theory f(x)")

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()




    