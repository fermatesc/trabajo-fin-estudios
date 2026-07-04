"""Pipeline end-to-end de entrenamiento de QGNN_Portfolio con múltiples semillas.

Entrena el modelo sobre FinancialGraphDataset para cada semilla de SEEDS,
guarda los pesos de cada modelo en qgnn_model_seed_<seed>.pth y produce una
gráfica comparativa de convergencia en plots/convergencia_training.png.
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.dataset_loader import FinancialGraphDataset
from models.qgnn_model import QGNN_Portfolio
from models.loss_functions import DifferentiableSharpeLoss
from config import set_global_seed, SEED, EPOCHS, LEARNING_RATE, IN_CHANNELS, HIDDEN_CHANNELS, N_QUBITS, Q_LAYERS, TRAIN_RATIO, VAL_RATIO

SEEDS = [42, 43, 44, 45, 46]


def train_and_evaluate():
    """Entrena y valida QGNN_Portfolio para cada semilla de SEEDS, guardando pesos y gráfica."""
    print("Iniciando pipeline de entrenamiento (End-to-End) con múltiples semillas...")

    dataset = FinancialGraphDataset(tau=0.5)
    total_samples = len(dataset)
    
    train_size = int(total_samples * TRAIN_RATIO)
    val_size = int(total_samples * VAL_RATIO)
    
    print(f"Entrenando sobre {train_size} ventanas temporales diarias y validando sobre {val_size}...")
    
    all_val_losses = {}
    
    for seed in SEEDS:
        print(f"\n{'='*40}")
        print(f"ENTRENANDO CON SEMILLA {seed}")
        print(f"{'='*40}")
        
        set_global_seed(seed)
        
        model = QGNN_Portfolio(in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS, n_qubits=N_QUBITS, q_layers=Q_LAYERS)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = DifferentiableSharpeLoss(risk_free_rate=0.0)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            
            for i in tqdm(range(train_size), desc=f"Epoch {epoch+1}/{EPOCHS}"):
                data = dataset[i]
                optimizer.zero_grad()
                weights = model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(weights, data.y, data.cov)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / train_size
            train_losses.append(avg_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in range(train_size, train_size + val_size):
                    data = dataset[i]
                    weights = model(data.x, data.edge_index, data.edge_attr)
                    loss = criterion(weights, data.y, data.cov)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / val_size
            val_losses.append(avg_val_loss)
            
            print(f"-> Epoch {epoch+1} Finalizada | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
        all_val_losses[seed] = val_losses

        model_path = f"qgnn_model_seed_{seed}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Pesos del modelo guardados en '{model_path}'.")

    plt.figure(figsize=(10, 5))
    for seed in SEEDS:
        plt.plot(range(1, EPOCHS + 1), all_val_losses[seed], marker='s', label=f'Val Loss (Seed {seed})')
    plt.title("Convergencia del Modelo Híbrido (QGNN) - Múltiples Semillas")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Negative Sharpe)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/convergencia_training.png")
    plt.close()
    print("\nGráfica de convergencia guardada en 'plots/convergencia_training.png'.")


if __name__ == "__main__":
    train_and_evaluate()
