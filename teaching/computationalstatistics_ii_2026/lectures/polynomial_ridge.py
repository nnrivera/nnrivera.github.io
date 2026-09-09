import numpy as np

# Same sample and degree as the unregularized comparison.
rng = np.random.default_rng(123)
n, p = 12, 10
x = rng.uniform(-2 * np.pi, 2 * np.pi, size=n)
y = np.sin(x) + rng.normal(0.0, 0.25, size=n)

# Design matrix: 1, x, ..., x^p; standardize only the nonconstant columns.
X = np.polynomial.polynomial.polyvander(x, p)
X_mean = X[:, 1:].mean(axis=0)
X_std = X[:, 1:].std(axis=0)
X[:, 1:] = (X[:, 1:] - X_mean) / X_std

# Match the matrix penalty 0.0001 used in polinomio.py.
lam = 0.0001 / n
P = np.eye(p + 1)
P[0, 0] = 0.0  # Do not penalize the intercept.

beta_ridge = np.linalg.solve(X.T @ X + n * lam * P, X.T @ y)
beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]

# Columns are the OLS and Ridge coefficients, in standardized coordinates.
print("     OLS        Ridge")
print(np.round(np.column_stack((beta_ols, beta_ridge)), 6))
print("Training MSE (OLS, Ridge):")
print(np.mean((y - X @ beta_ols) ** 2),
      np.mean((y - X @ beta_ridge) ** 2))
