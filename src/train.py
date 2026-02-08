"""
Training script for Animal vs Vehicle classifier on CIFAR-10.

Questo script implementa il training completo della CNN con:
- Split corretto train/validation/test (best practice ML)
- Monitoraggio di accuracy, precision, recall, F1 durante il training
- Early stopping opzionale per evitare overfitting
- Salvataggio del miglior modello basato su validation accuracy
"""

from __future__ import annotations
import json
import os
import sys
from typing import Dict, Tuple, Optional

# Aggiungi il path del progetto a sys.path per permettere gli import
# Questo permette di eseguire lo script da qualsiasi directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from models.cnn import SimpleCNN
from src.dataset import make_loaders
from src.utils import load_config, pick_device, set_seed, ensure_dirs


@torch.no_grad()
def evaluate_metrics(model: nn.Module, loader, device: torch.device) -> Tuple[float, float, float, float]:
    """
    Calcola metriche complete sul dataset di validazione/test.
    
    Args:
        model: Modello CNN da valutare
        loader: DataLoader con i dati da valutare
        device: Device su cui eseguire i calcoli (CPU/GPU)
        
    Returns:
        Tuple di (accuracy, precision, recall, f1_score)
        Le metriche sono calcolate considerando Vehicle (1) come classe positiva
    """
    model.eval()
    y_true = []
    y_pred = []
    
    for x, y, _cifar in loader:
        x = x.to(device)
        y = torch.tensor(y, device=device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(y.cpu().numpy().tolist())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calcola tutte le metriche
    # pos_label=1 significa che Vehicle è la classe positiva
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    return acc, prec, rec, f1


def train_one_epoch(
    model: nn.Module, 
    loader, 
    optimizer, 
    criterion, 
    device: torch.device, 
    log_every: int = 100
) -> float:
    """
    Addestra il modello per una epoca completa.
    
    Un'epoca significa processare tutti i batch del training set una volta.
    Per ogni batch:
    1. Calcola le predizioni del modello (forward pass)
    2. Calcola la loss (errore)
    3. Calcola i gradienti (backward pass)
    4. Aggiorna i pesi del modello (optimizer step)
    
    Args:
        model: Modello CNN da addestrare
        loader: DataLoader con i dati di training
        optimizer: Ottimizzatore (AdamW) per aggiornare i pesi
        criterion: Funzione di loss (CrossEntropyLoss)
        device: Device su cui eseguire i calcoli
        log_every: Ogni quanti step stampare la loss media
        
    Returns:
        Loss media dell'epoca
    """
    model.train()  # Imposta il modello in modalità training
    running_loss = 0.0

    for step, (x, y, _cifar) in enumerate(tqdm(loader, desc="Train", leave=False), start=1):
        # Sposta i dati sul device corretto (GPU se disponibile)
        x = x.to(device)
        y = torch.tensor(y, device=device)

        # Reset dei gradienti prima di ogni backward pass
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass: calcola le predizioni
        logits = model(x)
        
        # Calcola la loss (errore tra predizioni e verità)
        loss = criterion(logits, y)
        
        # Backward pass: calcola i gradienti
        loss.backward()
        
        # Aggiorna i pesi del modello usando i gradienti
        optimizer.step()

        running_loss += loss.item()

        # Stampa la loss ogni N step per monitorare il progresso
        if log_every and step % log_every == 0:
            avg = running_loss / step
            tqdm.write(f"step={step} avg_loss={avg:.4f}")

    return running_loss / max(len(loader), 1)


def main() -> None:
    """
    Funzione principale per il training del modello.
    
    Esegue il training completo seguendo queste fasi:
    1. Carica la configurazione e imposta il seed per riproducibilità
    2. Prepara i dataloader (train/val/test)
    3. Inizializza modello, loss e optimizer
    4. Loop di training per N epoche:
       - Addestra su training set
       - Valuta su validation set
       - Salva il miglior modello
       - Applica early stopping se configurato
    5. Salva la storia delle metriche
    """
    # Carica la configurazione dal file YAML
    config = load_config("configs/default.yaml")
    
    # Imposta il seed per garantire riproducibilità
    set_seed(int(config["seed"]))

    # Seleziona il device (GPU se disponibile, altrimenti CPU)
    device = pick_device(config["train"]["device"])
    print(f"[INFO] Using device: {device}")

    # Prepara le directory di output
    out_base = config["outputs"]["base_dir"]
    ckpt_path = config["outputs"]["checkpoint_path"]
    metrics_dir = config["outputs"]["metrics_dir"]
    ensure_dirs(out_base, os.path.dirname(ckpt_path), metrics_dir)

    # Crea i dataloader: train, validation e test
    # IMPORTANTE: Il test set NON viene usato durante il training!
    train_loader, val_loader, test_loader = make_loaders(
        root_dir=config["data"]["root_dir"],
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        val_split_ratio=float(config["data"]["val_split_ratio"]),
        seed=int(config["seed"]),
        device=config["train"]["device"],
    )

    # Inizializza il modello CNN
    model = SimpleCNN(dropout=float(config["model"]["dropout"])).to(device)
    print(f"[INFO] Model initialized: {config['model']['name']}")

    # Loss function: CrossEntropyLoss è standard per classificazione multi-classe
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: AdamW è una variante migliorata di Adam con weight decay corretto
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    # Variabili per tracciare il miglior modello e early stopping
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = config["train"].get("early_stopping_patience")
    
    # Storia completa delle metriche per ogni epoca
    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    num_epochs = int(config["train"]["epochs"])
    print(f"\n[INFO] Starting training for {num_epochs} epochs...")
    print(f"[INFO] Early stopping patience: {early_stopping_patience if early_stopping_patience else 'Disabled'}\n")

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"[INFO] Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # Fase 1: Training
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            log_every=int(config["train"]["log_every"])
        )
        
        # Fase 2: Valutazione sul validation set
        # Calcola anche la validation loss per early stopping
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _cifar in val_loader:
                x = x.to(device)
                y = torch.tensor(y, device=device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        
        # Calcola tutte le metriche sul validation set
        val_acc, val_prec, val_rec, val_f1 = evaluate_metrics(model, val_loader, device)

        # Salva le metriche nella storia
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_precision"].append(float(val_prec))
        history["val_recall"].append(float(val_rec))
        history["val_f1"].append(float(val_f1))

        # Stampa i risultati dell'epoca
        print(f"[RESULT] Train Loss: {train_loss:.4f}")
        print(f"[RESULT] Val Loss:   {val_loss:.4f}")
        print(f"[RESULT] Val Acc:    {val_acc:.4f}")
        print(f"[RESULT] Val Prec:   {val_prec:.4f}")
        print(f"[RESULT] Val Recall: {val_rec:.4f}")
        print(f"[RESULT] Val F1:     {val_f1:.4f}")

        # Salva il miglior modello basato su validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0  # Reset del contatore di patience
            
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"[INFO] ✓ Saved best checkpoint (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Patience: {patience_counter}/{early_stopping_patience}")

        # Early stopping: ferma il training se non c'è miglioramento
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"\n[INFO] Early stopping triggered after {epoch} epochs")
            print(f"[INFO] Best validation accuracy: {best_val_acc:.4f}")
            break

    # Valutazione finale sul test set (solo ora lo usiamo!)
    print(f"\n{'='*60}")
    print("[INFO] Final evaluation on TEST set (unseen data)...")
    print(f"{'='*60}")
    test_acc, test_prec, test_rec, test_f1 = evaluate_metrics(model, test_loader, device)
    print(f"[FINAL] Test Accuracy:  {test_acc:.4f}")
    print(f"[FINAL] Test Precision:  {test_prec:.4f}")
    print(f"[FINAL] Test Recall:     {test_rec:.4f}")
    print(f"[FINAL] Test F1-score:   {test_f1:.4f}")

    # Salva la storia completa delle metriche
    history["final_test_metrics"] = {
        "accuracy": float(test_acc),
        "precision": float(test_prec),
        "recall": float(test_rec),
        "f1": float(test_f1),
    }
    
    metrics_path = os.path.join(metrics_dir, "training_history.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n[INFO] Training history saved to {metrics_path}")
    print(f"[INFO] Best model saved to {ckpt_path}")
    print(f"[INFO] Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()