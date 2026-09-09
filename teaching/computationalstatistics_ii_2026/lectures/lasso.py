import numpy as np


def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


def kkt_error(X, y, beta, lam):
    c = X.T @ (y - X @ beta) / len(y)
    violation = np.where(
        beta != 0,
        np.abs(c - lam * np.sign(beta)),
        np.maximum(np.abs(c) - lam, 0.0),
    )
    return np.max(violation)


def lasso_cd(X, y, lam, beta0=None, tol=1e-7, max_sweeps=10000):
    """Minimize ||y - X beta||^2 / (2n) + lam * ||beta||_1."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if y.shape != (n,) or n == 0 or p == 0:
        raise ValueError("Use a nonempty X and a response vector y.")
    if lam < 0 or tol <= 0 or max_sweeps < 1:
        raise ValueError("Require lam >= 0, tol > 0, max_sweeps >= 1.")

    a = np.sum(X ** 2, axis=0) / n
    if np.any(a == 0):
        raise ValueError("Remove constant columns before centering/scaling.")
    beta = np.zeros(p) if beta0 is None else np.array(beta0, dtype=float, copy=True)
    r = y - X @ beta

    for sweep in range(1, max_sweeps + 1):
        for j in range(p):
            r += X[:, j] * beta[j]          # Restore the old contribution.
            z = X[:, j] @ r / n
            beta[j] = soft_threshold(z, lam) / a[j]
            r -= X[:, j] * beta[j]          # Subtract the new contribution.

        if kkt_error(X, y, beta, lam) <= tol:
            return beta, sweep

    raise RuntimeError("Lasso did not reach the requested KKT tolerance.")


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    n, p = 100, 20
    X_raw = rng.normal(size=(n, p))
    beta_true = np.r_[3.0, -2.0, 1.5, np.zeros(p - 3)]
    y_raw = 2.0 + X_raw @ beta_true + rng.normal(0.0, 1.5, n)

    x_mean = X_raw.mean(axis=0)
    x_scale = X_raw.std(axis=0)  # ddof=0 gives sum(X[:, j]**2) / n = 1.
    X = (X_raw - x_mean) / x_scale
    y_mean = y_raw.mean()
    y = y_raw - y_mean

    lam = 0.25
    beta, sweeps = lasso_cd(X, y, lam)
    print("Standardized coefficients:", np.round(beta, 3))
    print("Sweeps:", sweeps, "KKT error:", kkt_error(X, y, beta, lam))

    # Recover coefficients and predictions in the original units.
    beta_original = beta / x_scale
    intercept = y_mean - x_mean @ beta_original
    y_hat = intercept + X_raw @ beta_original
