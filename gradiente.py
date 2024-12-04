import numpy as np


# Define a função objetivo e o gradiente
def f(x):
    x1, x2, x3 = x
    return x1**2 + x1 * x3 - x2 + x2**2 + x2 * x3 + 3 * x3**2


def grad_f(x):
    x1, x2, x3 = x
    return np.array(
        [2 * x1 + x3, -1 + 2 * x2 + x3, x1 + x2 + 6 * x3]  # df/dx1  # df/dx2  # df/dx3
    )


# Parâmetros do método
alpha = 0.1  # Tamanho do passo
tolerance = 1e-4  # Tolerância
max_iter = 1  # Máximo de iterações

# Valor inicial
x = np.array([0.5, 0.1, 0.0])
history = [x]  # Histórico das iterações

# Método do Gradiente
for i in range(max_iter):
    grad = grad_f(x)
    grad_norm = np.linalg.norm(grad)

    # Verifica o critério de parada
    if grad_norm < tolerance:
        break

    # Atualiza a solução
    x = x - alpha * grad
    history.append(x)

# Resultados
final_value = f(x)
iterations = i + 1

print(f"x = {x}")
print(f"f(x) = {final_value}")
print(f"gn = {grad_norm}")
print(f"it = {iterations}")
