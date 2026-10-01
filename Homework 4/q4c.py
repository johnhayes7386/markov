import numpy as np
import matplotlib.pyplot as plt



np.random.seed(2340)

R = 10_000


P = np.array([
    [0,   1/2, 1/2, 0,   0],
    [1/4, 0,   0,   1/2, 1/4],
    [3/4, 0,   0,   0,   1/4],
    [0,   0,   0,   1,   0],
    [0,   0,   0,   0,   1]
])



def simulate(start):
    state = start
    t = 0

    while state != 3 and state != 4:

        u = np.random.uniform(0, 1)

        if state == 0:       # U
            if u < 1/2:
                state = 1    # I
            else:
                state = 2    # M

        elif state == 1:     # I
            if u < 1/4:
                state = 0    # U
            elif u < 3/4:
                state = 3    # F
            else:
                state = 4    # A

        elif state == 2:     # M
            if u < 3/4:
                state = 0    # U
            else:
                state = 4    # A

        t += 1

    return t, state


names = {
    0: "U",
    1: "I",
    2: "M"
}

all_results = {}

for start in [0, 1, 2]:

    times = []
    outcomes = []

    for i in range(R):
        t, outcome = simulate(start)

        times.append(t)
        outcomes.append(outcome)

    times = np.array(times)
    outcomes = np.array(outcomes)


    h_hat = np.mean(outcomes == 3)

  
    g_hat = np.mean(times)


    fold_times = times[outcomes == 3]
    aggregate_times = times[outcomes == 4]

    tau_F_hat = np.mean(fold_times)
    tau_A_hat = np.mean(aggregate_times)

    all_results[start] = {
        "times": times,
        "outcomes": outcomes,
        "h": h_hat,
        "g": g_hat,
        "tau_F": tau_F_hat,
        "tau_A": tau_A_hat
    }

    print("--------------------------------")
    print("Starting state:", names[start])
    print("h_hat =", h_hat)
    print("g_hat =", g_hat)
    print("tau_F_hat =", tau_F_hat)
    print("tau_A_hat =", tau_A_hat)




print("\n\nCOMPARISON WITH EXACT VALUES")
print("--------------------------------")

exact_h = {
    0: 1/2,
    1: 5/8,
    2: 3/8
}

exact_g = {
    0: 4,
    1: 5/2,
    2: 4
}

exact_tau_F = {
    0: 4,
    1: 9/5,
    2: 5
}

exact_tau_A = {
    0: 4,
    1: 7/3,
    2: 17/5
}

for start in [0, 1, 2]:

    result = all_results[start]

    print("\nStarting state:", names[start])

    print("h:")
    print("  Simulation =", result["h"])
    print("  Exact      =", exact_h[start])

    print("g:")
    print("  Simulation =", result["g"])
    print("  Exact      =", exact_g[start])

    print("tau_F:")
    print("  Simulation =", result["tau_F"])
    print("  Exact      =", exact_tau_F[start])

    print("tau_A:")
    print("  Simulation =", result["tau_A"])
    print("  Exact      =", exact_tau_A[start])




times_I = all_results[1]["times"]
outcomes_I = all_results[1]["outcomes"]

fold_times_I = times_I[outcomes_I == 3]
aggregate_times_I = times_I[outcomes_I == 4]




plt.figure(figsize=(9, 6))

plt.hist(
    fold_times_I,
    bins=np.arange(0.5, max(fold_times_I) + 1.5, 1),
    density=True,
    edgecolor="black"
)

plt.xlabel("Time to fold")
plt.ylabel("Probability")
plt.title("Starting from I: Conditional Folding Time")
plt.grid(axis="y", alpha=0.3)

plt.show()




plt.figure(figsize=(9, 6))

plt.hist(
    aggregate_times_I,
    bins=np.arange(0.5, max(aggregate_times_I) + 1.5, 1),
    density=True,
    edgecolor="black"
)

plt.xlabel("Time to aggregate")
plt.ylabel("Probability")
plt.title("Starting from I: Conditional Aggregation Time")
plt.grid(axis="y", alpha=0.3)

plt.show()


print("\n\nCHECKING g_x RELATIONSHIP")
print("--------------------------------")

for start in [0, 1, 2]:

    h = exact_h[start]
    tau_F = exact_tau_F[start]
    tau_A = exact_tau_A[start]

    calculated_g = h * tau_F + (1 - h) * tau_A

    print(
        names[start],
        ":",
        calculated_g,
        "  Exact g =",
        exact_g[start]
    )