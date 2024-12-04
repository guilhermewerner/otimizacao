import numpy as np


# Função para montar a tabela inicial do Simplex
def montar_tabela(c, A, b):
    num_variaveis = A.shape[1]
    num_restricoes = A.shape[0]

    # Adicionar variáveis de folga
    I = np.eye(num_restricoes)
    tabela = np.hstack((A, I, b.reshape(-1, 1)))

    # Linha Z (função objetivo)
    z_row = np.hstack((-c, np.zeros(num_restricoes + 1)))

    # Combinar tabela com a linha Z
    tabela = np.vstack((tabela, z_row))
    return tabela


# Função para executar uma iteração do Simplex
def simplex_iteration(tabela):
    # Critério de entrada (menor valor negativo na linha Z)
    z_row = tabela[-1, :-1]
    coluna_pivo = np.argmin(z_row)
    if z_row[coluna_pivo] >= 0:
        return tabela, True  # Solução ótima alcançada

    # Critério de saída (razão b[i] / tabela[i, coluna_pivo] para valores > 0)
    coluna = tabela[:-1, coluna_pivo]
    b = tabela[:-1, -1]

    # Evitar divisão por zero explicitamente
    with np.errstate(divide="ignore", invalid="ignore"):
        razoes = np.where(coluna > 0, b / coluna, np.inf)

    linha_pivo = np.argmin(razoes)
    if razoes[linha_pivo] == np.inf:
        raise ValueError(
            "Problema não possui solução limitada."
        )  # Caso não haja pivô válido

    # Atualizar a tabela (pivoteamento)
    pivo = tabela[linha_pivo, coluna_pivo]
    tabela[linha_pivo] /= pivo

    for i in range(tabela.shape[0]):
        if i != linha_pivo:
            tabela[i] -= tabela[i, coluna_pivo] * tabela[linha_pivo]

    return tabela, False


# Método Simplex
def simplex(c, A, b, max_iter=100):
    tabela = montar_tabela(c, A, b)
    history = [tabela.copy()]

    for i in range(max_iter):
        tabela, is_optimal = simplex_iteration(tabela)
        history.append(tabela.copy())
        if is_optimal:
            break

    # Resultados finais
    solucao = np.zeros(len(c))
    for i in range(len(c)):
        col = tabela[:-1, i]
        if np.count_nonzero(col) == 1 and np.sum(col) == 1:
            solucao[i] = tabela[
                np.where(col == 1)[0][0], -1
            ]  # Extrair o valor escalar explicitamente

    valor_objetivo = -tabela[-1, -1]
    return {
        "solucao": solucao,
        "valor_objetivo": valor_objetivo,
        "iteracoes": i + 1,
        "history": history,
    }


# Parâmetros do problema
c = np.array([12, 18, 22])  # Função objetivo
A = np.array([[1.5, 0, 1.2], [0, 2.2, 1.4], [1.2, 2, 2.4]])
b = np.array([120, 200, 250])

# Resolver o problema com Simplex
resultado = simplex(c, A, b)

# Exibir resultados
print(f"Solução ótima: {resultado['solucao']}")
print(f"Valor ótimo da função objetivo: {resultado['valor_objetivo']}")
print(f"Número de iterações: {resultado['iteracoes']}")
