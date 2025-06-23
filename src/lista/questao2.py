import numpy as np

# Função objetivo
def f(x):
    return 5*x**2 + np.exp(3*x)

# Gradiente da função objetivo
def grad_f(x):
    return 10*x + 3*np.exp(3*x)

# Hessiana da função objetivo
def hess_f(x):
    return 10 + 9*np.exp(3*x)

# Método de Newton-Raphson
def newton_raphson(f, grad_f, hess_f, x0, tolerance=1e-4, max_iter=1000):
    x = x0
    for i in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)

        if abs(grad) < tolerance:
            print(f"Convergência atingida após {i+1} iterações.")
            return x, f(x)

        if hess == 0:
            raise ValueError("Hessiana nula, método não pode continuar.")

        x = x - grad / hess

    print("Número máximo de iterações atingido.")
    return x, f(x)

# Ponto inicial e parâmetros iniciais
x0 = 0.0  # Ponto inicial

# Resolver o problema
resultado, valor_objetivo = newton_raphson(f, grad_f, hess_f, x0)

# Exibir resultados
print(f"Solução encontrada: {resultado}")
print(f"Valor da função objetivo: {valor_objetivo}")
