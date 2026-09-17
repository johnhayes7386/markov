import numpy as np

P = np.array([
    [9/10, 1/10, 0],
    [0,    3/4,  1/4],
    [1/2,  0,    1/2]
])

P50 = np.linalg.matrix_power(P, 50)

print(P50)