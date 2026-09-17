import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# ==========================================
# SETTINGS
# ==========================================

R = 20_000
T = 10_000
d0 = 10

rng = np.random.default_rng(42)

times = np.arange(T + 1)


# ==========================================
# PART 4(b): ONE LION
# ==========================================

D1 = np.full(R, d0, dtype=int)
alive1 = np.ones(R, dtype=bool)

S1 = np.zeros(T + 1)
S1[0] = 1.0

for t in range(1, T + 1):

    indices = np.where(alive1)[0]

    changes = rng.choice(
        [-2, 0, 2],
        size=len(indices),
        p=[0.25, 0.50, 0.25]
    )

    D1[indices] += changes

    captured = indices[D1[indices] == 0]

    alive1[captured] = False

    S1[t] = np.mean(alive1)


# ==========================================
# FIT BETA 1
# ==========================================

fit_mask1 = (
    (times >= 100) &
    (times <= 10_000) &
    (S1 > 0)
)

slope1, intercept1 = np.polyfit(
    np.log(times[fit_mask1]),
    np.log(S1[fit_mask1]),
    1
)

beta1 = -slope1

print(f"beta_1 = {beta1:.4f}")


# ==========================================
# PART 4(c): TWO LIONS
# ==========================================

# Positions of lamb and two lions
lamb = np.zeros(R, dtype=int)

lion1 = np.full(R, d0, dtype=int)
lion2 = np.full(R, d0, dtype=int)

alive2 = np.ones(R, dtype=bool)

S2 = np.zeros(T + 1)
S2[0] = 1.0


for t in range(1, T + 1):

    indices = np.where(alive2)[0]

    # Each animal independently moves -1 or +1
    lamb_move = rng.choice(
        [-1, 1],
        size=len(indices)
    )

    lion1_move = rng.choice(
        [-1, 1],
        size=len(indices)
    )

    lion2_move = rng.choice(
        [-1, 1],
        size=len(indices)
    )

    # Update positions
    lamb[indices] += lamb_move
    lion1[indices] += lion1_move
    lion2[indices] += lion2_move

    # Capture occurs if either lion reaches the lamb
    captured_mask = (
        (lion1[indices] == lamb[indices]) |
        (lion2[indices] == lamb[indices])
    )

    captured = indices[captured_mask]

    alive2[captured] = False

    S2[t] = np.mean(alive2)


# ==========================================
# FIT BETA 2
# ==========================================

fit_mask2 = (
    (times >= 100) &
    (times <= 10_000) &
    (S2 > 0)
)

slope2, intercept2 = np.polyfit(
    np.log(times[fit_mask2]),
    np.log(S2[fit_mask2]),
    1
)

beta2 = -slope2

print(f"beta_2 = {beta2:.4f}")


# ==========================================
# TABLE VALUES
# ==========================================

print("\nComparison:")
print("t\tS2(t)\t\tS1(t)^2")

for t in [100, 1000, 10000]:

    print(
        f"{t}\t"
        f"{S2[t]:.6f}\t"
        f"{S1[t]**2:.6f}"
    )


# ==========================================
# CONTINUUM PREDICTION FOR 4(b)
# ==========================================

t_positive = times[1:]

S1_theory = erf(
    d0 / (2 * np.sqrt(t_positive))
)


# ==========================================
# PLOT EVERYTHING ON SAME FIGURE
# ==========================================

plt.figure(figsize=(9, 7))

# One lion
plt.loglog(
    t_positive,
    S1[1:],
    label=rf"$S_1(t)$, $\beta_1={beta1:.3f}$"
)

# Theory from part b
plt.loglog(
    t_positive,
    S1_theory,
    "--",
    label=r"$\mathrm{erf}(d_0/(2\sqrt{t}))$"
)

# Two lions
plt.loglog(
    t_positive,
    S2[1:],
    label=rf"$S_2(t)$, $\beta_2={beta2:.3f}$"
)

# What survival would look like if independent
plt.loglog(
    t_positive,
    S1[1:]**2,
    "--",
    label=r"$S_1(t)^2$"
)

plt.xlabel("Time t")
plt.ylabel("Survival Probability")

plt.title(
    r"Lamb Survival, fit window $10^2 \leq t \leq 10^4$"
)

plt.legend()

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.tight_layout()

plt.show()