"""Genera el conjunto de gráficos de análisis exploratorio de datos (EDA) del
dataset de precios/retornos/correlaciones, y los guarda como PNG en `plots/`.

Produce: figura1_precios_normalizados.png, figura2_histograma_retornos.png,
figura3_heatmap_correlacion.png, figura4_correlacion_dinamica.png,
figura5_topologia_grafo.png, figura6_distribucion_correlaciones.png,
figura7_volatilidad_movil.png, figura8_grafo_estatico_sectores.png,
figura9_similitud_dinamica.png.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import networkx as nx


def main():
    """Carga los datos de `data/` y genera las 9 figuras EDA en `plots/`.

    Notes:
        En la Figura 6, al aplanar `dynamic_correlations.npy` se filtran los
        valores >= 0.999 para excluir la diagonal principal (autocorrelación
        = 1) del histograma de correlaciones. En la Figura 9, el índice
        temporal de la similitud dinámica arranca en `returns.index[59:]`
        porque `dynamic_distances.npy` se calculó con una ventana móvil de
        60 días.
    """
    data_dir = "data"
    plots_dir = "plots"

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.dpi'] = 150

    os.makedirs(plots_dir, exist_ok=True)

    prices = pd.read_csv(os.path.join(data_dir, "prices_adjusted.csv"), index_col=0, parse_dates=True)
    returns = pd.read_csv(os.path.join(data_dir, "returns_log.csv"), index_col=0, parse_dates=True)

    volatility = pd.read_csv(os.path.join(data_dir, "volatility_rolling.csv"), index_col=0, parse_dates=True)
    dynamic_corr = np.load(os.path.join(data_dir, "dynamic_correlations.npy"))

    wiki_matrix = np.load(os.path.join(data_dir, "static_wikipedia_sectors.npy"))
    dynamic_dist = np.load(os.path.join(data_dir, "dynamic_distances.npy"))

    print("Iniciando generación de gráficos EDA...")

    print("Generando Figura 1: Precios Normalizados...")
    plt.figure(figsize=(12, 6))
    sample_assets = ['AAPL', 'JPM', 'JNJ', 'XOM', 'WMT']

    normalized_prices = prices[sample_assets] / prices[sample_assets].iloc[0] * 100
    
    for col in normalized_prices.columns:
        plt.plot(normalized_prices.index, normalized_prices[col], label=col, linewidth=1.5)
    
    plt.title("Figura 1: Evolución de Precios Normalizados (Base 100)", fontsize=14, weight='bold')
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Precio Normalizado", fontsize=12)
    plt.legend(title="Activo", loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figura1_precios_normalizados.png"), dpi=300)
    plt.close()

    print("Generando Figura 2: Histograma de Retornos (Fat Tails)...")
    plt.figure(figsize=(10, 6))

    all_returns = returns.values.flatten()
    all_returns = all_returns[~np.isnan(all_returns)]

    mu, std = norm.fit(all_returns)

    plt.hist(all_returns, bins=100, density=True, alpha=0.6, color='steelblue', label='Retornos Empíricos S&P 500')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label=rf'Distribución Normal Teórica\n($\mu$={mu:.4f}, $\sigma$={std:.4f})')

    plt.title("Figura 2: Distribución de Retornos Logarítmicos Diarios", fontsize=14, weight='bold')
    plt.xlabel("Retorno Diario", fontsize=12)
    plt.ylabel("Densidad de Frecuencia", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.xlim(-0.15, 0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figura2_histograma_retornos.png"), dpi=300)
    plt.close()

    print("Generando Figura 3: Mapa de Calor de Correlación...")
    plt.figure(figsize=(12, 10))

    corr_matrix = returns.corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot_kws={"size": 8})
    
    plt.title("Figura 3: Matriz de Correlación de Pearson de los Activos Seleccionados", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figura3_heatmap_correlacion.png"), dpi=300)
    plt.close()

    print("Generando Figura 4: Evolución de Correlación Dinámica...")
    plt.figure(figsize=(12, 6))

    corr_pair = returns['AAPL'].rolling(window=60).corr(returns['XOM']).dropna()
    
    plt.plot(corr_pair.index, corr_pair, label='Correlación AAPL vs XOM (60 días)', color='purple', linewidth=1.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    
    plt.title("Figura 4: Evolución de la Correlación Móvil (Ventana 60 días) entre Tech y Energía", fontsize=14, weight='bold')
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Correlación de Pearson", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figura4_correlacion_dinamica.png"), dpi=300)
    plt.close()

    print("Generando Figura 5: Topología del Grafo...")
    plt.figure(figsize=(10, 10))

    corr_matrix_t = dynamic_corr[-1]
    n_assets = len(returns.columns)

    tau = 0.5

    G = nx.Graph()
    assets = returns.columns

    for i in range(n_assets):
        G.add_node(assets[i])
        for j in range(i + 1, n_assets):
            if abs(corr_matrix_t[i, j]) > tau:
                weight = abs(corr_matrix_t[i, j])
                G.add_edge(assets[i], assets[j], weight=weight)

    pos = nx.spring_layout(G, k=0.5, seed=42)
    edges = G.edges(data=True)
    weights = [d['weight'] * 5 for (u, v, d) in edges]
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', alpha=0.5)
    
    plt.title(f"Figura 5: Topología del Grafo G(t) (Umbral $\\tau$={tau})", fontsize=14, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figura5_topologia_grafo.png"), dpi=300)
    plt.close()

    print("Generando Figura 6: Distribución de Correlaciones...")
    plt.figure(figsize=(10, 6))

    all_corrs = dynamic_corr.flatten()
    all_corrs = all_corrs[all_corrs < 0.999]

    sns.histplot(all_corrs, bins=100, kde=True, color='teal')
    plt.axvline(0.5, color='red', linestyle='--', label=r'Posible Umbral ($\tau$=0.5)')
    plt.axvline(-0.5, color='red', linestyle='--')
    
    plt.title("Figura 6: Distribución Histórica de todas las Correlaciones Móviles", fontsize=14, weight='bold')
    plt.xlabel("Valor de Correlación de Pearson", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figura6_distribucion_correlaciones.png"), dpi=300)
    plt.close()

    print("Generando Figura 7: Evolución de la Volatilidad...")
    plt.figure(figsize=(12, 6))

    sample_assets = ['AAPL', 'JPM', 'JNJ', 'XOM', 'WMT']

    for col in sample_assets:
        plt.plot(volatility.index, volatility[col], label=col, linewidth=1.5)
        
    plt.title("Figura 7: Volatilidad Móvil (Desviación Estándar a 60 días)", fontsize=14, weight='bold')
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Volatilidad", fontsize=12)
    plt.legend(title="Activo", loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figura7_volatilidad_movil.png"), dpi=300)
    plt.close()

    print("Generando Figura 8: Grafo Estático (Sectores GICS)...")
    plt.figure(figsize=(10, 10))

    G_wiki = nx.Graph()
    assets = returns.columns
    n_assets = len(assets)

    for i in range(n_assets):
        G_wiki.add_node(assets[i])
        for j in range(i + 1, n_assets):
            if wiki_matrix[i, j] == 1.0:
                G_wiki.add_edge(assets[i], assets[j], weight=1.0)

    pos_wiki = nx.spring_layout(G_wiki, k=0.8, seed=42)
    
    nx.draw_networkx_nodes(G_wiki, pos_wiki, node_color='lightgreen', node_size=800, alpha=0.8)
    nx.draw_networkx_labels(G_wiki, pos_wiki, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G_wiki, pos_wiki, edge_color='gray', alpha=0.6, style='dashed')
    
    plt.title("Figura 8: Grafo Estático (Conexiones por Sector GICS de Wikipedia)", fontsize=14, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figura8_grafo_estatico_sectores.png"), dpi=300)
    plt.close()

    print("Generando Figura 9: Evolución de Similitud Dinámica...")
    plt.figure(figsize=(12, 6))

    idx_aapl = list(assets).index('AAPL')
    idx_msft = list(assets).index('MSFT')

    sim_aapl_msft = dynamic_dist[:, idx_aapl, idx_msft]

    time_index = returns.index[60-1:]
    
    plt.plot(time_index, sim_aapl_msft, label='Similitud AAPL vs MSFT (Euclidiana Inversa)', color='darkorange', linewidth=1.5)
    
    plt.title("Figura 9: Evolución de la Similitud Dinámica (Ventana 60 días) - Sector Tech", fontsize=14, weight='bold')
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Similitud (1 / (1 + Distancia))", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "figura9_similitud_dinamica.png"), dpi=300)
    plt.close()

    print(f"¡Éxito! Todas las gráficas en alta resolución se han guardado en '{plots_dir}'.")


if __name__ == "__main__":
    main()
