"""
Funzioni helper per l'interfaccia Streamlit.

Questo modulo contiene funzioni di utilità per:
- Caricamento del modello con caching
- Predizione su immagini
- Caricamento metriche e dati
"""

from __future__ import annotations
import os
import sys
import json
import torch
from typing import Tuple, Dict, Any, Optional
from PIL import Image
from torchvision import transforms

# Aggiungi il path del progetto a sys.path per permettere gli import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.cnn import SimpleCNN
from src.utils import load_config, pick_device

LABELS = {0: "Animal", 1: "Vehicle"}

# Parametri di normalizzazione CIFAR-10
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def build_infer_transform() -> transforms.Compose:
    """
    Crea le trasformazioni per l'inference.
    Deve corrispondere alle trasformazioni usate durante il training.
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def load_model(ckpt_path: str, device: torch.device, dropout: float) -> Tuple[SimpleCNN, Dict[str, Any]]:
    """
    Carica il modello dal checkpoint.
    
    Args:
        ckpt_path: Path al file checkpoint
        device: Device su cui caricare il modello
        dropout: Valore dropout del modello
        
    Returns:
        Tuple di (modello, info_checkpoint)
    """
    model = SimpleCNN(dropout=dropout).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    info = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "best_val_acc": checkpoint.get("best_val_acc", checkpoint.get("best_acc", "unknown")),
        "best_val_loss": checkpoint.get("best_val_loss", "unknown"),
    }
    
    return model, info


@torch.no_grad()
def predict_image_streamlit(
    model: SimpleCNN, 
    device: torch.device, 
    image: Image.Image
) -> Tuple[str, float, float]:
    """
    Classifica un'immagine PIL e restituisce label, probabilità e confidence.
    
    Args:
        model: Modello CNN addestrato
        device: Device su cui eseguire la predizione
        image: Immagine PIL da classificare
        
    Returns:
        Tuple di (label, prob_animal, prob_vehicle)
        - label: "Animal" o "Vehicle"
        - prob_animal: Probabilità che sia un animale (0-1)
        - prob_vehicle: Probabilità che sia un veicolo (0-1)
    """
    transform = build_infer_transform()
    
    # Converti in RGB se necessario
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Applica trasformazioni
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predizione
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1)
    
    prob_animal = probs[0][0].item()
    prob_vehicle = probs[0][1].item()
    pred = int(torch.argmax(logits, dim=1).item())
    
    label = LABELS[pred]
    
    return label, prob_animal, prob_vehicle


def load_metrics(metrics_path: str) -> Optional[Dict[str, Any]]:
    """
    Carica le metriche di valutazione dal file JSON.
    
    Args:
        metrics_path: Path al file evaluation.json
        
    Returns:
        Dizionario con le metriche o None se il file non esiste
    """
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_training_history(history_path: str) -> Optional[Dict[str, list]]:
    """
    Carica la storia del training dal file JSON.
    
    Args:
        history_path: Path al file training_history.json
        
    Returns:
        Dizionario con la storia del training o None se il file non esiste
    """
    if not os.path.exists(history_path):
        return None
    
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_error_summary(summary_path: str) -> Optional[Dict[str, Any]]:
    """
    Carica le statistiche degli errori dal file JSON.
    
    Args:
        summary_path: Path al file summary.json nella cartella misclassified
        
    Returns:
        Dizionario con le statistiche degli errori o None se il file non esiste
    """
    if not os.path.exists(summary_path):
        return None
    
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_misclassified_images(misclassified_dir: str, max_images: int = 30) -> list:
    """
    Ottiene la lista delle immagini classificate erroneamente.
    
    Args:
        misclassified_dir: Directory contenente le immagini errate
        max_images: Numero massimo di immagini da restituire
        
    Returns:
        Lista di path alle immagini
    """
    if not os.path.exists(misclassified_dir):
        return []
    
    image_files = [
        os.path.join(misclassified_dir, f)
        for f in os.listdir(misclassified_dir)
        if f.endswith(('.png', '.jpg', '.jpeg')) and f.startswith('err_')
    ]
    
    # Ordina per nome (che contiene il numero)
    image_files.sort()
    
    return image_files[:max_images]
