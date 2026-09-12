import matplotlib.pyplot as plt
import numpy as np

# Simulate the sample and choose the polynomial degree.
rng = np.random.default_rng(123)
n, p = 12, 10
x = rng.uniform(-2 * np.pi, 2 * np.pi, size=n)
y = np.sin(x) + rng.normal(0.0, 0.25, size=n)

# Design matrix: 1, x, ..., x^p; standardise only the nonconstant columns.
X = np.polynomial.polynomial.polyvander(x, p)
X_mean = X[:, 1:].mean(axis=0)
X_std = X[:, 1:].std(axis=0)
X[:, 1:] = (X[:, 1:] - X_mean) / X_std

# Fit ordinary least squares.
beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
y_hat = X @ beta_ols

# Coefficients are in standardised coordinates; beta_ols[0] is the intercept.
print("OLS coefficients:")
print(np.round(beta_ols, 6))
print("Training MSE (OLS):", np.mean((y - y_hat) ** 2))

# Build a prediction grid using the sample means and scales.
x_grid = np.linspace(x.min(), x.max(), 500)
X_grid = np.polynomial.polynomial.polyvander(x_grid, p)
X_grid[:, 1:] = (X_grid[:, 1:] - X_mean) / X_std

# Display the fitted curve.
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="gray", label="Sample", zorder=3)
plt.plot(x_grid, np.sin(x_grid), "--", color="purple", label="True mean")
plt.plot(x_grid, X_grid @ beta_ols, color="blue", label="OLS")
plt.title(f"Polynomial OLS regression, degree {p}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()
