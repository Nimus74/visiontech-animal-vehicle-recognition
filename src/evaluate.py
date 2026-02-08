"""
Evaluation script:
- Carica il miglior checkpoint salvato durante il training
- Calcola Accuracy, Precision, Recall, F1 sul test set
- Stampa Confusion Matrix e Classification Report
- Salva le metriche in formato JSON per analisi successive
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Aggiungi il path del progetto a sys.path per permettere gli import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.cnn import SimpleCNN
from src.dataset import make_loaders
from src.utils import load_config, pick_device, ensure_dirs


def main() -> None:
    config = load_config("configs/default.yaml")
    device = pick_device(config["train"]["device"])

    ckpt_path = config["outputs"]["checkpoint_path"]
    metrics_dir = config["outputs"]["metrics_dir"]
    ensure_dirs(metrics_dir)

    # Carica i dataloader: train, val, test
    # Per la valutazione finale usiamo solo il test set
    _, _, test_loader = make_loaders(
        root_dir=config["data"]["root_dir"],
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        val_split_ratio=float(config["data"]["val_split_ratio"]),
        seed=int(config["seed"]),
        device=config["train"]["device"],
    )

    model = SimpleCNN(dropout=float(config["model"]["dropout"])).to(device)

    # Carica il checkpoint del miglior modello
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}. Esegui prima il training!")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"[INFO] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"[INFO] Best validation accuracy: {checkpoint.get('best_val_acc', 'unknown'):.4f}")

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y, _cifar in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(list(y))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)   # Vehicle as positive class
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(y_true, y_pred, target_names=["Animal(0)", "Vehicle(1)"], digits=4)

    print("\n[METRICS]")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f} (Vehicle=positive)")
    print(f"Recall   : {rec:.4f} (Vehicle=positive)")
    print(f"F1-score : {f1:.4f} (Vehicle=positive)")

    print("\n[CONFUSION MATRIX] rows=true, cols=pred")
    print(cm)

    print("\n[CLASSIFICATION REPORT]")
    print(report)

    out = {
        "accuracy": float(acc),
        "precision_vehicle": float(prec),
        "recall_vehicle": float(rec),
        "f1_vehicle": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    out_path = os.path.join(metrics_dir, "evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n[INFO] Saved evaluation metrics to {out_path}")


if __name__ == "__main__":
    main()