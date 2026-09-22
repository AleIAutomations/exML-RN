import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Configurações da Simulação
tamanho_amostra = 30    # 'n': tamanho de cada amostra (30 é a regra de bolso do TLC)
num_frames = 100         # Duração da animação
amostras_por_frame = 15  # Quantas amostras são coletadas por quadro

# 2. População Original (Distribuição Exponencial)
# Uma distribuição que claramente não é normal para mostrar o TLC em ação
populacao = np.random.exponential(scale=2.0, size=100000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot Fixo da População Original
ax1.hist(populacao, bins=50, color='skyblue', density=True)
ax1.set_title("População Original (Exponencial)")
ax1.set_xlim(0, 10)

# 3. Setup do Gráfico Animado (Distribuição das Médias)
medias = []
ax2.set_xlim(0.5, 4)
ax2.set_ylim(0, 1.5)
ax2.set_title(f"Distribuição das Médias (n={tamanho_amostra})")

def init():
    ax2.clear()
    ax2.set_xlim(0.5, 4)
    ax2.set_ylim(0, 1.5)
    return []

def update(frame):
    # Sorteia novas amostras da população e calcula a média de cada uma
    for _ in range(amostras_por_frame):
        amostra = np.random.choice(populacao, size=tamanho_amostra)
        medias.append(np.mean(amostra))
    
    ax2.clear()
    
    # Plota o histograma atualizado das médias
    ax2.hist(medias, bins=40, color='salmon', density=True, range=(0.5, 4))
    ax2.set_title(f"Distribuição das Médias (n={tamanho_amostra})\nTotal de Amostras: {len(medias)}")
    ax2.set_xlim(0.5, 4)
    ax2.set_ylim(0, 1.5)
    
    # Desenha a Curva Normal Teórica por cima (linha tracejada preta)
    if len(medias) > 10:
        mu = 2.0 # Média teórica da exponencial
        sigma = 2.0 / np.sqrt(tamanho_amostra) # Erro padrão
        x = np.linspace(0.5, 4, 100)
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
        ax2.plot(x, y, 'k--', linewidth=2)

# 4. Criar e Exibir a Animação
ani = animation.FuncAnimation(
    fig, update, frames=num_frames, init_func=init, interval=100, repeat=False
)

plt.tight_layout()
plt.show()