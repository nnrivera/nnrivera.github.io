import matplotlib.pyplot as plt
import numpy as np

# Simulate the sample and choose the polynomial degree.
rng = np.random.default_rng(123)
n, p = 12, 10
x = rng.uniform(-2 * np.pi, 2 * np.pi, size=n)
y = np.sin(x) + rng.normal(0.0, 0.25, size=n)

# Design matrix: 1, x, ..., x^p; standardize only the nonconstant columns.
X = np.polynomial.polynomial.polyvander(x, p)
X_mean = X[:, 1:].mean(axis=0)
X_std = X[:, 1:].std(axis=0)
X[:, 1:] = (X[:, 1:] - X_mean) / X_std

# Fit Lasso: ||y - X beta||^2 / (2n) + lam * sum(abs(beta[1:])).
beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
lambdas = [0.6, 0.1, 0.01, 0.001]  # Strongest penalty first.
tol = 1e-8
max_sweeps = 200000  # Correlated polynomial columns can converge slowly.
a = np.mean(X ** 2, axis=0)
beta = np.r_[y.mean(), np.zeros(p)]
fits = {}

for lam in lambdas:
    # Reuse the previous solution as the starting point.
    r = y - X @ beta
    for sweep in range(1, max_sweeps + 1):
        # Update the intercept without penalizing it.
        delta = r.mean()
        beta[0] += delta
        r -= delta

        # Update each slope by soft thresholding.
        for j in range(1, p + 1):
            r += X[:, j] * beta[j]
            z = X[:, j] @ r / n
            beta[j] = np.sign(z) * max(abs(z) - lam, 0.0) / a[j]
            r -= X[:, j] * beta[j]

        # Check the optimality conditions, including the intercept.
        c = X.T @ (y - X @ beta) / n
        violation = np.empty(p + 1)
        violation[0] = abs(c[0])
        violation[1:] = np.where(
            beta[1:] != 0,
            np.abs(c[1:] - lam * np.sign(beta[1:])),
            np.maximum(np.abs(c[1:]) - lam, 0.0),
        )
        if np.max(violation) <= tol:
            break
    else:
        raise RuntimeError("Lasso did not converge; increase max_sweeps.")

    fits[lam] = beta.copy()
    y_hat = X @ beta
    print(f"lambda={lam:g}: sweeps={sweep}, Training MSE={np.mean((y - y_hat) ** 2):.6f}")
    print("Zero slope indices:", np.flatnonzero(beta[1:] == 0) + 1)

# Columns are the OLS and Lasso coefficients, in standardized coordinates.
print("Columns: OLS, then Lasso with lambdas", sorted(lambdas))
coefficients = np.column_stack([beta_ols] + [fits[lam] for lam in sorted(lambdas)])
print(np.round(coefficients, 3))

# Build a prediction grid using the sample means and scales.
x_grid = np.linspace(x.min(), x.max(), 500)
X_grid = np.polynomial.polynomial.polyvander(x_grid, p)
X_grid[:, 1:] = (X_grid[:, 1:] - X_mean) / X_std

# Display one fitted curve per penalty.
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
for ax, lam in zip(axes.flat, sorted(lambdas)):
    beta_lasso = fits[lam]
    ax.scatter(x, y, color="gray", label="Sample", zorder=3)
    ax.plot(x_grid, np.sin(x_grid), "--", color="purple", label="True mean")
    ax.plot(x_grid, X_grid @ beta_lasso, color="red", label="Lasso")
    count = np.count_nonzero(beta_lasso[1:])
    ax.set_title(f"lambda={lam:g}: {count} nonzero slopes")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
axes[0, 0].legend()
fig.suptitle(f"Polynomial Lasso regression, degree {p}")
plt.tight_layout()
plt.show()
