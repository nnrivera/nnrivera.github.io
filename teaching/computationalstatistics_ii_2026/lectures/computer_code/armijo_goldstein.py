# Compare constant-step gradient descent with the Armijo-Goldstein rule.
# Requires NumPy. Run this file from top to bottom.

import numpy as np

np.random.seed(2026)
X = np.c_[np.ones((100, 1)), 2 * np.random.rand(100, 1)]
# a simple Model!
Y = 4 + 3 * X[:, 1:] + np.random.randn(100, 1)
n = len(Y)
# ==========================================
# 2. Define Cost and Gradient
# ==========================================
def cost(theta):
    residuals = Y - X.dot(theta)
    return (1 / (2 * n)) * np.sum(residuals**2)

def gradient(theta):
    return (1 / n) * X.T.dot(X.dot(theta) - Y)

# theta_init[0] is b (intercept), theta_init[1] is a (slope)
theta_init = np.random.randn(2, 1)
n_iterations = 25

# ==========================================
# Strategy 1: Constant Step (gamma = 0.05)
# ==========================================
theta_c = theta_init.copy()
cost_c = [cost(theta_c)]

for _ in range(n_iterations):
    theta_c = theta_c - 0.05 * gradient(theta_c)
    cost_c.append(cost(theta_c))

# ==========================================
# Strategy 2: Armijo-Goldstein Rule
# ==========================================
alpha = 0.1
beta = 0.9

theta_g = theta_init.copy()
cost_g = [cost(theta_g)]

for _ in range(n_iterations):
    g = gradient(theta_g)
    c = cost(theta_g)

    gamma = 1.0
    low = 0.0
    high = float('inf')

    while True:
        theta_new = theta_g - gamma * g
        c_new = cost(theta_new)

        actual_drop = c - c_new
        predicted_drop = gamma * np.sum(g**2)

        if actual_drop < alpha * predicted_drop:
            high = gamma
            gamma = (low + high) / 2.0
        elif actual_drop > beta * predicted_drop:
            low = gamma
            if high == float('inf'):
                gamma *= 2.0
            else:
                gamma = (low + high) / 2.0
        else:
            break

    theta_g = theta_g - gamma * g
    cost_g.append(cost(theta_g))

# ==========================================
# 3. Print Comparison
# ==========================================
print("--- Gradient Descent Race (25 Iterations) ---")
print(f"Initial Cost:             {cost_c[0]:.4f}")
print(f"Final Cost (Constant):    {cost_c[-1]:.4f}")
print(f"Final Cost (Goldstein):   {cost_g[-1]:.4f}\n")

print("--- Final Model Parameters ---")
print("True Parameters:   b (intercept) = 4.00, a (slope) = 3.00")
print(f"Constant Step:     b = {theta_c[0][0]:.2f}, a = {theta_c[1][0]:.2f}")
print(f"Goldstein Step:    b = {theta_g[0][0]:.2f}, a = {theta_g[1][0]:.2f}")
