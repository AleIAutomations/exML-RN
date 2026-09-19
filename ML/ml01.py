## exercicio de regressao linear
## Você foi contratado por uma imobiliária local. Eles querem prever o preço de venda de casas com base no tamanho do imóvel em metros quadrados.
## Objetivo: Implementar a regressão linear com o método do Gradiente Descendente (Gradient Descent) para encontrar a reta ideal de previsão.

#--------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# 1 - preparar os dados 
# tamaho em m² (X) e Preço em milhares de R$ (y)

X = np.array([30,50,70,90,110,130], dtype=float)
y = np.array([150,230,310,400,480,560], dtype=float)

# normalaizacao simples para estabilidade do gradiente descendente 
X_mean, X_std = X.mean(), X.std()
X_scaled = (X - X_mean) / X_std

# 2 - definicao dos parametros e hipotese
w = 0.0  # peso 
b = 0.0  # vies

learning_rate = 0.1
epochs = 100

# 3 - loops de treinamento
loss_history = []
for epochs in range(epochs):

    # previsão linear
    y_pred = w * X_scaled + b

    # calculo do erro quadrático médio
    error = y_pred - y
    loss = np.mean(error ** 2)
    loss_history.append(loss)

    # derivados parciais
    dw = (2 / len(X)) * np.dot(error, X_scaled)
    db = (2 / len(X)) * np.sum(error) 

    # atualizacao de parametros
    w -= learning_rate * dw
    w -= learning_rate * db

# 4 - previsao e visualizacao
print(f"treinamento concluido! Peso w: {w:.2f}, Vies b:{b:.2f}")

# prever preco pra uma casa de 80 m²
m2_novo = 80
m2_novo_scaled = (m2_novo - X_mean) / X_std
preco_previsao = w * m2_novo_scaled + b 
print(f"Preço previsto para 80m²: R$ {preco_previsao * 1000:.2f}")

# visualiazao da linha de refressao
plt.figure(figsize=(8, 4))
plt.scatter(X, y, color="blue", label="Dados reais (Casas)")
plt.plot(X, w * X_scaled + b, color= "red", label="Reta de Regressao")
plt.xlabel("Tamanho (m²)")
plt.xlabel("Preço (Milhares de R$)")
plt.title("Regressao Linear - Preço de Imovéis")
plt.legend()
plt.grid(True)
plt.show()






