"""
Script per visualizzare i risultati del training e della valutazione.

Genera grafici per:
- Training curves (loss e accuracy nel tempo)
- Confusion matrix visualizzata
- Confronto metriche tra epoche
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# Aggiungi il path del progetto a sys.path per permettere gli import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_config, ensure_dirs


# Configurazione per grafici più belli
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    # Fallback per versioni più vecchie di matplotlib
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")


def plot_training_curves(history: Dict[str, list], save_path: str) -> None:
    """
    Crea grafici delle curve di training.
    
    Mostra l'andamento di:
    - Training loss e Validation loss
    - Validation accuracy, precision, recall, F1
    
    Args:
        history: Dizionario con le metriche per ogni epoca
        save_path: Path dove salvare il grafico
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Crea una figura con 2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Subplot 1: Loss curves
    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, history["val_loss"], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoca', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training e Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Accuracy curve
    ax2 = axes[1]
    ax2.plot(epochs, history["val_acc"], 'g-o', label='Validation Accuracy', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoca', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Training curves saved to {save_path}")
    plt.close()


def plot_metrics_comparison(history: Dict[str, list], save_path: str) -> None:
    """
    Crea un grafico comparativo di tutte le metriche di validazione.
    
    Args:
        history: Dizionario con le metriche per ogni epoca
        save_path: Path dove salvare il grafico
    """
    epochs = range(1, len(history["val_acc"]) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plotta tutte le metriche insieme
    ax.plot(epochs, history["val_acc"], 'g-o', label='Accuracy', linewidth=2, markersize=5)
    ax.plot(epochs, history["val_precision"], 'b-s', label='Precision', linewidth=2, markersize=5)
    ax.plot(epochs, history["val_recall"], 'r-^', label='Recall', linewidth=2, markersize=5)
    ax.plot(epochs, history["val_f1"], 'm-d', label='F1-Score', linewidth=2, markersize=5)
    
    ax.set_xlabel('Epoca', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Confronto Metriche di Validazione', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Metrics comparison saved to {save_path}")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: str, class_names: list = None) -> None:
    """
    Crea una visualizzazione della confusion matrix.
    
    La confusion matrix mostra:
    - Quante immagini sono state classificate correttamente (diagonale)
    - Quante immagini sono state classificate erroneamente (fuori diagonale)
    
    Args:
        cm: Matrice di confusione (2x2 per classificazione binaria)
        save_path: Path dove salvare il grafico
        class_names: Nomi delle classi (default: ["Animal", "Vehicle"])
    """
    if class_names is None:
        class_names = ["Animal", "Vehicle"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Crea heatmap della confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Numero di campioni'}
    )
    
    ax.set_xlabel('Predizione', fontsize=12, fontweight='bold')
    ax.set_ylabel('Verità', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Aggiungi annotazioni per spiegare la matrice
    # True Negatives (Animal classificato come Animal)
    tn = cm[0, 0]
    # False Positives (Animal classificato come Vehicle)
    fp = cm[0, 1]
    # False Negatives (Vehicle classificato come Animal)
    fn = cm[1, 0]
    # True Positives (Vehicle classificato come Vehicle)
    tp = cm[1, 1]
    
    # Aggiungi testo esplicativo
    textstr = f'True Negatives (TN): {tn}\nFalse Positives (FP): {fp}\nFalse Negatives (FN): {fn}\nTrue Positives (TP): {tp}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Confusion matrix saved to {save_path}")
    plt.close()


def main() -> None:
    """
    Funzione principale per generare tutte le visualizzazioni.
    
    Legge i file JSON con le metriche e genera i grafici corrispondenti.
    """
    config = load_config("configs/default.yaml")
    metrics_dir = config["outputs"]["metrics_dir"]
    plots_dir = config["outputs"]["plots_dir"]
    
    ensure_dirs(plots_dir)
    
    # Carica la storia del training
    history_path = os.path.join(metrics_dir, "training_history.json")
    if not os.path.exists(history_path):
        print(f"[WARNING] File {history_path} non trovato. Esegui prima il training!")
        return
    
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    # Genera i grafici delle curve di training
    plot_training_curves(history, os.path.join(plots_dir, "training_curves.png"))
    
    # Genera il confronto delle metriche
    if "val_precision" in history:
        plot_metrics_comparison(history, os.path.join(plots_dir, "metrics_comparison.png"))
    
    # Carica e visualizza la confusion matrix se disponibile
    eval_path = os.path.join(metrics_dir, "evaluation.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        if "confusion_matrix" in eval_data:
            cm = np.array(eval_data["confusion_matrix"])
            plot_confusion_matrix(cm, os.path.join(plots_dir, "confusion_matrix.png"))
    
    print(f"\n[INFO] Tutte le visualizzazioni sono state salvate in {plots_dir}")


if __name__ == "__main__":
    main()
