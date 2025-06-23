import numpy as np
from scipy.optimize import minimize


# Função objetivo
def f(x):
    x1, x2, x3 = x
    return x1**2 + x1 * x3 - x2 + x2**2 + x2 * x3 + 3 * x3**2


# Gradiente da função objetivo
def grad_f(x):
    x1, x2, x3 = x
    df_dx1 = 2 * x1 + x3
    df_dx2 = -1 + 2 * x2 + x3
    df_dx3 = x1 + x2 + 6 * x3
    return np.array([df_dx1, df_dx2, df_dx3])


# Restrições
# x1 + x1*x3 + x2 <= 10
# 3*x1 + x2 = 5
constraints = [
    {"type": "ineq", "fun": lambda x: 10 - (x[0] + x[0] * x[2] + x[1])},
    {"type": "eq", "fun": lambda x: 3 * x[0] + x[1] - 5},
]

# Ponto inicial
x0 = np.array([1.0, 1.0, 1.0])

# Resolver o problema com o método do Gradiente
resultado = minimize(
    f, x0, jac=grad_f, constraints=constraints, tol=1e-4, method="SLSQP"
)

# Exibir resultados
if resultado.success:
    print("Convergência atingida.")
else:
    print("Otimização falhou.")

print(f"Solução encontrada: {resultado.x}")
print(f"Valor da função objetivo: {resultado.fun}")
