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