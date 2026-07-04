"""Pipeline end-to-end de entrenamiento de ClassicalGNN_Portfolio con múltiples semillas.

Entrena el modelo clásico (gemelo del QGNN, sin componente cuántico) sobre
FinancialGraphDataset para cada semilla de SEEDS y guarda los pesos de cada
modelo en classical_gnn_model_seed_<seed>.pth.
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.dataset_loader import FinancialGraphDataset
from models.classical_gnn_model import ClassicalGNN_Portfolio
from models.loss_functions import DifferentiableSharpeLoss
from config import set_global_seed, SEED, EPOCHS, LEARNING_RATE, IN_CHANNELS, HIDDEN_CHANNELS, TRAIN_RATIO, VAL_RATIO

SEEDS = [42, 43, 44, 45, 46]


def train_and_evaluate():
    """Entrena y valida ClassicalGNN_Portfolio para cada semilla de SEEDS, guardando los pesos.

    Notes:
        `latent_dim=4` es el parámetro equivalente a `n_qubits` en el modelo
        cuántico; se fija a 4 para igualar aproximadamente el número de
        parámetros entre ambos modelos y hacer la comparación justa.
    """
    print("Iniciando pipeline de entrenamiento (End-to-End) Classical GNN con múltiples semillas...")

    dataset = FinancialGraphDataset(tau=0.5)
    total_samples = len(dataset)
    
    train_size = int(total_samples * TRAIN_RATIO)
    val_size = int(total_samples * VAL_RATIO)
    
    print(f"Entrenando sobre {train_size} ventanas temporales diarias y validando sobre {val_size}...")
    
    all_val_losses = {}
    
    for seed in SEEDS:
        print(f"\n{'='*40}")
        print(f"ENTRENANDO CLASSICAL GNN CON SEMILLA {seed}")
        print(f"{'='*40}")
        
        set_global_seed(seed)

        model = ClassicalGNN_Portfolio(in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS, latent_dim=4)
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

        model_path = f"classical_gnn_model_seed_{seed}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Pesos del modelo guardados en '{model_path}'.")


if __name__ == "__main__":
    train_and_evaluate()
