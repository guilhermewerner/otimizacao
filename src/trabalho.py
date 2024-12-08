import numpy as np
from scipy.optimize import minimize

print("----")

for r in range(1, 6):

    def objetivo(x):
        x1, x2 = x
        return -(r * x1 + (x1 * x2) / 2 + r**2)

    def restricao(x):
        x1, x2 = x
        return np.pi * r**2 - (2 * r * x1 + x1 * x2)

    x0 = [0.5, 0.5]

    restricoes = {"type": "ineq", "fun": restricao}
    bounds = [(0, r), (0, 0.53 * 2 * r)]

    resultado = minimize(objetivo, x0, constraints=restricoes, bounds=bounds)

    print(f"r  = {r}")

    if resultado.success:
        print(f"x1 = {resultado.x[0]}")
        print(f"x2 = {resultado.x[1]}")
        print(f"S  = {resultado.fun * -1}")
    else:
        print("O problema não convergiu")

    print("----")
