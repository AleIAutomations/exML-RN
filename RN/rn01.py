# O Perceptron de Rosenblatt (Porta Lógica AND)
# Você está construindo o componente lógico fundamental para um novo processador
# Implementar a menor unidade de uma rede neural: o Perceptron de camada única, ensinando-o a resolver a 
# tabela verdade da operação lógica


import numpy as np
import matplotlib.pyplot as plt

# 1. Dados da Tabela Verdade AND
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# 2. Inicialização do Perceptron
weights = np.random.randn(2)
bias = 0.0
lr = 0.1
epochs = 20

# 3. Treinamento (Regra do Perceptron)
for epoch in range(epochs):
    for i in range(len(X)):
        # Soma Ponderada
        linear_output = np.dot(X[i], weights) + bias
        # Ativação Degrau (Step Function)
        y_pred = 1 if linear_output >= 0 else 0
        
        # Erro
        error = y[i] - y_pred
        
        # Atualização dos Pesos e Viés
        weights += lr * error * X[i]
        bias += lr * error

print(f"Pesos Aprendidos: {weights}, Viés: {bias}")

# Teste
for x_input in X:
    pred = 1 if (np.dot(x_input, weights) + bias) >= 0 else 0
    print(f"Entrada: {x_input} -> Saída Prevista: {pred}")


# Explicação Didática da Lógica e Palavras-Chave:Soma Ponderada: $z = (w_1 \cdot x_1 + w_2 \cdot x_2) + b$. 
# Multiplica-se cada entrada pelo seu peso respectivo e soma-se o viés.Ativação Degrau: Retorna 1 se o valor da soma 
# linear for maior ou igual a zero, e 0 caso contrário.Regra do Perceptron: Se a previsão estiver correta (error = 0), 
# nenhum parâmetro muda. Se estiver incorreta, atualiza-se a direção do vetor de pesos.