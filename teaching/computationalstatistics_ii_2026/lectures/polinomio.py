import matplotlib.pyplot as plt
import numpy as np


def main():
    # Simulamos una muestra (x_i, y_i) con una relación no lineal.
    rng = np.random.default_rng(123)
    n = 10

    x = rng.uniform(-2 * np.pi, 2 * np.pi, size=n)
    y = np.sin(x) + rng.normal(loc=0.0, scale=0.25, size=n)

    # Matriz de diseño:
    # columnas = 1, x, x^2, ..., x^p
    p = 9
    X = np.polynomial.polynomial.polyvander(x, p)

    # Estandarizamos cada columna excepto el intercepto.
    X_mean = X[:, 1:].mean(axis=0)
    X_std = X[:, 1:].std(axis=0)

    X_standardized = X.copy()
    X_standardized[:, 1:] = (
        X[:, 1:] - X_mean
    ) / X_std

    # Parámetro de regularización Ridge.
    lambda_ridge = 0.0001

    # Matriz de penalización.
    # El primer elemento es cero para no penalizar el intercepto.
    penalty = np.eye(p + 1)
    penalty[0, 0] = 0.0

    # Estimador Ridge:
    # beta = (X'X + lambda * P)^(-1) X'y
    #
    # Usamos solve() en lugar de calcular explícitamente la inversa.
    beta_ridge = np.linalg.solve(
        X_standardized.T @ X_standardized
        + lambda_ridge * penalty,
        X_standardized.T @ y,
    )

    print(
        f"Coeficientes Ridge para un polinomio de grado {p} "
        f"con lambda = {lambda_ridge}:"
    )
    for j, beta_j in enumerate(beta_ridge):
        print(f"beta_{j} = {beta_j:.6f}")

    # Construimos la matriz de diseño para la grilla.
    x_grid = np.linspace(x.min(), x.max(), 500)
    X_grid = np.polynomial.polynomial.polyvander(x_grid, p)

    # Aplicamos la estandarización calculada con la muestra.
    X_grid_standardized = X_grid.copy()
    X_grid_standardized[:, 1:] = (
        X_grid[:, 1:] - X_mean
    ) / X_std

    # Predicciones Ridge.
    y_hat_ridge = X_grid_standardized @ beta_ridge

    # Gráfico.
    plt.clf()

    plt.scatter(
        x,
        y,
        color="gray",
        alpha=0.7,
        label="Muestra",
    )

   

    plt.plot(
        x_grid,
        y_hat_ridge,
        color="red",
        linewidth=2,
        label=f"Ridge ($\\lambda={lambda_ridge}$)",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Regresión polinómica Ridge de grado {p}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()