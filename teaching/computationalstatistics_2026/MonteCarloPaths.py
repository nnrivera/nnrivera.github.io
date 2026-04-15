import numpy as np
from math import comb

# Generate a random NE path in an n x n grid: 0 = North, 1 = East
def random_path(n):
  path = np.zeros(2 * n, dtype=int)
  path[np.random.choice(2 * n, n, replace=False)] = 1
  return path

# Generate N random NE paths
def random_paths(n, N):
  return [random_path(n) for _ in range(N)]

# Count number of corners (i.e., direction changes)
def corners(p):
  pp = p[1:]
  return np.sum(np.abs(p[:-1] - pp))

# Monte Carlo approximation of number of NE paths with ≤ ell corners
def monte_carlo_paths(n, N, ell):
  paths = random_paths(n, N)
  num_within_ell = sum(corners(p) <= ell for p in paths)
  total_paths = comb(2 * n, n)
  return total_paths * num_within_ell / N

# Test
n = 20  # grid size
ell = 8 # max number of corners
N = 1000000  # number of samples

result = monte_carlo_paths(n, N, ell)
print(result)
