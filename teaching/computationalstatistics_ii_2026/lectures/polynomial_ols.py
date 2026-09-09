import numpy as np

# Simulate the sample.
rng = np.random.default_rng(123)
n = 12
x = rng.uniform(-2 * np.pi, 2 * np.pi, size=n)
y = np.sin(x) + rng.normal(0.0, 0.25, size=n)

# Choose the polynomial degree.
p = 4
X = np.polynomial.polynomial.polyvander(x, p)

# Standardize all columns except the intercept.
X_mean = X[:, 1:].mean(axis=0)
X_std = X[:, 1:].std(axis=0)
X[:, 1:] = (X[:, 1:] - X_mean) / X_std

# Fit ordinary least squares and compute fitted values.
beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
y_hat = X @ beta_ols

print(beta_ols)
