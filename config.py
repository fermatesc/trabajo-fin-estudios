"""Configuración centralizada de hiperparámetros del TFM.

Todos los módulos deben importar desde aquí para garantizar consistencia.
Agrupa los parámetros en bloques: reproducibilidad (semilla global), datos
(tickers, rango temporal, rutas de datos y de gráficas), grafo (ventana
móvil y umbral de correlación para construir las aristas), modelo QGNN
(dimensiones de entrada/ocultas, qubits y capas del VQC, temperatura de la
softmax, dropout), entrenamiento (tasa de aprendizaje, épocas, particiones
train/val/test y tasa libre de riesgo) y backtesting (coste de transacción
y penalización anti-rotación de la loss `TurnoverAwareSharpeLoss`).

`TICKERS` agrupa 20 activos en 5 sectores (4 tickers cada uno, en este
orden): tecnología, finanzas, salud, energía y consumo.

`RISK_FREE_RATE = 0.045` corresponde a la tasa libre de riesgo anualizada
del T-bill a 3 meses, media de 2023 (fuente: FRED, serie DTB3).
"""
import random

import numpy as np
import torch

SEED = 42


def set_global_seed(seed: int = SEED):
    """Fija todas las semillas para reproducibilidad completa."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'NVDA',
    'JPM', 'BAC', 'GS', 'MS',
    'JNJ', 'PFE', 'UNH', 'ABBV',
    'XOM', 'CVX', 'COP', 'SLB',
    'PG', 'WMT', 'KO', 'PEP'
]
START_DATE = "2019-01-01"
END_DATE = "2024-01-01"
DATA_DIR = "data"
PLOTS_DIR = "plots"

ROLLING_WINDOW = 60
CORRELATION_THRESHOLD = 0.5

IN_CHANNELS = 6
HIDDEN_CHANNELS = 8
N_QUBITS = 4
Q_LAYERS = 2
SOFTMAX_TEMPERATURE = 5.0
DROPOUT_RATE = 0.2

LEARNING_RATE = 0.01
EPOCHS = 15
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RISK_FREE_RATE = 0.045

TRANSACTION_COST_BPS = 10

TURNOVER_PENALTY_LAMBDA = 0.0015
FIXED_TEMPERATURE = 1.5
