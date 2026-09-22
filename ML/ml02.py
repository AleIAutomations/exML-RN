# Um hospital quer automatizar a triagem de pacientes para determinar se um tumor é Benigno (0) ou Maligno (1) com base no tamanho do nódulo
# Programar a Regressão Logística usando a função de ativação Sigmoide e a função de custo Binary Cross-Entropy

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.0, 1.5, 2.0, 3.0, 3.5, 4.5, 5.0, 6.0, 6.5, 7.0])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = 0.0
b = 0.0
lr = 0.5
epochs = 500

for epochs in range(epochs):
    z = w * x + b

    y_hat = sigmoid(z)

    cost = -np.mean(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))

    dw = np.mean((y_hat - y) * x)
    db = np.mean(y_hat - y)

    w -= lr * dw
    b -= lr * db

X_test = np.linspace(0, 8, 100)
y_prob = sigmoid(w * X_test + b)

plt.figure(figsize=(8, 4)) 
plt.scatter(x, y, c=y, cmap="bwr", edgecolor="k", label = "Paciente")
plt.plot(X_test, y_prob, color="green", label ="Probabilidade Sigmoide")
plt.axhline(0.5, color="grey", linestyle = "--", label = "limiar de Decição (0.5)")
plt.xlabel("Tamanho do Tumor (mm)")
plt.ylabel("Probabilidade de ser Maligno")
plt.title("Classificação de Tumores com Regressão Logistica")
plt.legend()
plt.grid(True)
plt.show()