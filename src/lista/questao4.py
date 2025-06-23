import numpy as np
from scipy.optimize import minimize
from scipy.optimize import dual_annealing

# Dados fornecidos
# X e Y representam as variáveis independentes e dependentes, respectivamente.
X = np.array([19, 2, 9, 4, 5, 6, 3, 11, 14, 17, 1, 20])
Y = np.array([8, 25, 13, 17, 20, 13, 18, 8, 9, 6, 42, 7])


# Modelo de regressão não linear: Y = a * exp(b * X) + c
def modelo(params, X):
    a, b, c = params
    return a * np.exp(b * X) + c


# Função objetivo para ajustar os parâmetros a, b, c
def erro_quadratico(params):
    predicoes = modelo(params, X)
    return np.sum((Y - predicoes) ** 2)  # Soma dos erros quadráticos


# Gradiente numérico da função objetivo (para métodos baseados em gradiente)
def gradiente_erro_quadratico(params):
    epsilon = 1e-5  # Perturbação para derivada numérica
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_fwd = params.copy()
        params_bwd = params.copy()
        params_fwd[i] += epsilon
        params_bwd[i] -= epsilon
        grad[i] = (erro_quadratico(params_fwd) - erro_quadratico(params_bwd)) / (
            2 * epsilon
        )
    return grad


# Ponto inicial para os parâmetros
guess = [1.0, 0.0, 0.0]

# Resolução utilizando o método do Gradiente (SLSQP)
resultado_gradiente = minimize(
    erro_quadratico, guess, jac=gradiente_erro_quadratico, method="SLSQP", tol=1e-4
)

# Resolução utilizando Simulated Annealing (Dual Annealing)
resultado_annealing = dual_annealing(
    erro_quadratico, bounds=[(-10, 10), (-1, 1), (-10, 10)]
)

# Comparação dos resultados
print("\nResultados do Método do Gradiente:")
if resultado_gradiente.success:
    print("Convergência atingida.")
else:
    print("Otimização falhou.")
print(f"Parâmetros encontrados: {resultado_gradiente.x}")
print(f"Erro quadrático: {resultado_gradiente.fun}")

print("\nResultados do Simulated Annealing:")
print(f"Parâmetros encontrados: {resultado_annealing.x}")
print(f"Erro quadrático: {resultado_annealing.fun}")

# Análise dos métodos:
# 1. Método do Gradiente:
#    - Utiliza o gradiente para encontrar um mínimo local.
#    - É mais rápido para convergir, especialmente em problemas bem condicionados.
#    - Pode falhar em encontrar o mínimo global se a função tiver múltiplos mínimos locais.
#
# 2. Simulated Annealing:
#    - Método heurístico inspirado no recozimento metálico.
#    - Mais robusto para encontrar o mínimo global, mesmo em funções com muitos mínimos locais.
#    - Requer mais tempo computacional devido à natureza estocástica.
#
# Resultado esperado:
# O Simulated Annealing pode fornecer um ajuste mais globalmente ótimo, enquanto o método do Gradiente
# é útil para refinamento local rápido. A escolha do método depende do trade-off entre robustez e desempenho.
