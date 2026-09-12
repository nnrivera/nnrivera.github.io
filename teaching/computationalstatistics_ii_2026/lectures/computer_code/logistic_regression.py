# Compare logistic regression using gradient descent, scikit-learn, and statsmodels.
# Requires NumPy, pandas, scikit-learn, and statsmodels.
# Run this file from top to bottom; no other lecture files are needed.

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression


def sigmoid(z):
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


def logistic_nll(beta, X, y):
    """Average negative log-likelihood."""
    eta = X @ beta
    return np.mean(np.logaddexp(0.0, eta) - y * eta)


def logistic_gradient_descent(X, y, beta0=None,
                              learning_rate=None,
                              tol=1e-8, max_iter=100_000):
    n, p = X.shape
    beta = np.zeros(p) if beta0 is None else np.array(beta0, dtype=float)

    # Hessian <= X.T @ X / (4n), so 1/L is a safe constant step.
    if learning_rate is None:
        L = 0.25 * np.linalg.eigvalsh(X.T @ X / n).max()
        learning_rate = 1.0 / L

    history = []
    for iteration in range(max_iter):
        probabilities = sigmoid(X @ beta)
        gradient = X.T @ (probabilities - y) / n
        history.append(logistic_nll(beta, X, y))

        if np.linalg.norm(gradient, ord=2) < tol:
            break

        beta = beta - learning_rate * gradient
    else:
        raise RuntimeError("Gradient descent did not converge")

    return beta, np.asarray(history), iteration + 1


rng = np.random.default_rng(2026)
n = 2_000

# ----- Simulate data from a correctly specified logistic model -----
features = rng.normal(size=(n, 3))
features = (features - features.mean(axis=0)) / features.std(axis=0)
X = np.column_stack((np.ones(n), features))       # explicit intercept
beta_true = np.array([-0.40, 1.25, -0.80, 0.55])
y = rng.binomial(1, sigmoid(X @ beta_true))

# ----- 1. Our gradient-descent estimator -----
beta_gd, loss_history, gd_iterations = logistic_gradient_descent(
    X, y, tol=1e-10, max_iter=200_000
)

# ----- 2. scikit-learn: explicitly turn regularisation off -----
sk_fit = LogisticRegression(
    C=np.inf,              # zero penalty in current scikit-learn
    solver="lbfgs",
    fit_intercept=False,     # X already contains the intercept column
    tol=1e-12,
    max_iter=10_000,
)
sk_fit.fit(X, y)
beta_sklearn = sk_fit.coef_.ravel()

# ----- 3. statsmodels: Logit is unregularised by default -----
sm_fit = sm.Logit(y, X).fit(
    method="newton", disp=False, maxiter=200, tol=1e-12
)
beta_statsmodels = np.asarray(sm_fit.params)

# ----- Compare coefficients and fitted objectives -----
estimates = pd.DataFrame(
    {
        "truth": beta_true,
        "our_GD": beta_gd,
        "sklearn": beta_sklearn,
        "statsmodels": beta_statsmodels,
    },
    index=["intercept", "x1", "x2", "x3"],
)

def accuracy(beta):
    predictions = (sigmoid(X @ beta) >= 0.5).astype(int)
    return np.mean(predictions == y)

diagnostics = pd.DataFrame(
    {
        "average_NLL": [
            logistic_nll(beta_gd, X, y),
            logistic_nll(beta_sklearn, X, y),
            logistic_nll(beta_statsmodels, X, y),
        ],
        "accuracy": [
            accuracy(beta_gd),
            accuracy(beta_sklearn),
            accuracy(beta_statsmodels),
        ],
    },
    index=["our_GD", "sklearn", "statsmodels"],
)

print(estimates.round(6))
print(diagnostics.round(8))
print("GD iterations:", gd_iterations)
print(
    "largest coefficient difference:",
    np.ptp(
        estimates[["our_GD", "sklearn", "statsmodels"]].to_numpy(),
        axis=1,
    ).max(),
)
print(sm_fit.summary())
