import numpy as np

rng = np.random.default_rng(42)

U = rng.random(100_000)

z = 1 - np.sqrt(1-U)

print("Sample mean:", np.mean(z))
print("Theoretical mean:", 1/3)