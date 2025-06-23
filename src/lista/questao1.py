import numpy as np


# Função objetivo
def f(x):
    return x[0] ** 2 + x[0] * x[2] - x[1] + x[1] ** 2 + x[1] * x[2] + 3 * x[2] ** 2


# Gradiente da função objetivo
def grad_f(x):
    df_dx1 = 2 * x[0] + x[2]
    df_dx2 = 2 * x[1] + x[2] - 1
    df_dx3 = x[0] + x[1] + 6 * x[2]
    return np.array([df_dx1, df_dx2, df_dx3])


# Método do Gradiente
def gradient_descent(f, grad_f, x0, learning_rate=0.01, tolerance=1e-4, max_iter=1000):
    x = x0
    for i in range(max_iter):
        grad = grad_f(x)
        norm_grad = np.linalg.norm(grad)

        if norm_grad < tolerance:
            print(f"Convergência atingida após {i+1} iterações.")
            return x, f(x)

        x = x - learning_rate * grad

    print("Número máximo de iterações atingido.")
    return x, f(x)


# Ponto inicial e parâmetros iniciais
x0 = np.array([1.0, 1.0, 1.0])  # Ponto inicial
learning_rate = 0.01

# Resolver o problema
resultado, valor_objetivo = gradient_descent(f, grad_f, x0, learning_rate)

# Exibir resultados
print(f"Solução encontrada: {resultado}")
print(f"Valor da função objetivo: {valor_objetivo}")
