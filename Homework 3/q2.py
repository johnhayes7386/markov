import numpy as np

P = np.array([
    [1/2, 1/2, 0,   0,   0, 0],
    [1/3, 0,   1/3, 0, 1/3, 0],
    [0,   0,   1/4, 3/4, 0, 0],
    [0,   0,   1,   0,   0, 0],
    [0,   0,   0,   0,   0, 1],
    [0,   0,   0,   0,   1, 0]
])

P20 = np.linalg.matrix_power(P, 20)
P21 = np.linalg.matrix_power(P, 21)

np.set_printoptions(precision=3, suppress=True)

print("P^20:")
print(P20)

print("\nP^21:")
print(P21)