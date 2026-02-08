"""
Error analysis:
- Salva un campione di immagini classificate erroneamente in outputs/misclassified/
- Denormalizza le immagini per visualizzarle correttamente
- Analizza pattern di confusione sistematici (es. deer vs truck per pattern di sfondo simili)

Questo script aiuta a capire:
- Quali tipi di immagini vengono confuse più spesso
- Se ci sono pattern sistematici negli errori
- Come migliorare il modello in futuro
"""

from __future__ import annotations
import os
import sys
import json
from collections import Counter
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

# Aggiungi il path del progetto a sys.path per permettere gli import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.cnn import SimpleCNN
from src.dataset import make_loaders
from src.utils import load_config, pick_device, ensure_dirs

# Nomi delle classi CIFAR-10 per riferimento
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Parametri di normalizzazione CIFAR-10 (stessi usati nel training)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def denormalize(tensor: torch.Tensor, mean: tuple = CIFAR_MEAN, std: tuple = CIFAR_STD) -> torch.Tensor:
    """
    Denormalizza un tensore di immagine.
    
    Durante il training, le immagini vengono normalizzate (sottraendo la media
    e dividendo per la deviazione standard). Per visualizzarle correttamente,
    dobbiamo invertire questa operazione.
    
    Args:
        tensor: Tensore normalizzato [C, H, W] o [B, C, H, W]
        mean: Media usata per la normalizzazione
        std: Deviazione standard usata per la normalizzazione
        
    Returns:
        Tensore denormalizzato nello stesso formato
    """
    # Converte tuple in tensori
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    
    # Se il tensore ha dimensione batch, aggiusta la view
    if tensor.dim() == 4:
        mean_tensor = mean_tensor.view(1, -1, 1, 1)
        std_tensor = std_tensor.view(1, -1, 1, 1)
    
    # Denormalizza: x = (x_norm * std) + mean
    return tensor * std_tensor + mean_tensor


def create_error_grid(images: list, labels: list, preds: list, save_path: str, n_cols: int = 8) -> None:
    """
    Crea una griglia visuale delle immagini errate.
    
    Args:
        images: Lista di tensori immagine denormalizzati
        labels: Lista di label vere
        preds: Lista di predizioni
        save_path: Path dove salvare la griglia
        n_cols: Numero di colonne nella griglia
    """
    # Prepara le immagini per la griglia
    grid_images = []
    for img, true_label, pred_label in zip(images, labels, preds):
        # Clamp i valori tra 0 e 1 per la visualizzazione
        img_clamped = torch.clamp(img, 0, 1)
        grid_images.append(img_clamped)
    
    if not grid_images:
        print("[WARNING] Nessuna immagine da visualizzare nella griglia")
        return
    
    # Crea la griglia
    grid = make_grid(grid_images[:min(len(grid_images), 32)], nrow=n_cols, padding=2, pad_value=1.0)
    
    # Salva la griglia
    save_image(grid, save_path)
    print(f"[INFO] Error grid saved to {save_path}")


def main() -> None:
    """
    Funzione principale per l'analisi degli errori.
    
    Analizza le immagini classificate erroneamente dal modello e:
    1. Salva le immagini denormalizzate per visualizzazione
    2. Crea una griglia visuale degli errori
    3. Analizza i pattern di confusione più comuni
    4. Salva statistiche dettagliate
    """
    config = load_config("configs/default.yaml")
    device = pick_device(config["train"]["device"])

    ckpt_path = config["outputs"]["checkpoint_path"]
    out_dir = config["outputs"]["misclassified_dir"]
    max_to_save = config["outputs"].get("max_misclassified_images", 200)
    
    ensure_dirs(out_dir)

    # Verifica che il checkpoint esista
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}. Esegui prima il training!")

    # Carica i dataloader (usa solo il test set per l'analisi finale)
    _, _, test_loader = make_loaders(
        root_dir=config["data"]["root_dir"],
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        val_split_ratio=float(config["data"]["val_split_ratio"]),
        seed=int(config["seed"]),
        device=config["train"]["device"],
    )

    # Carica il modello
    model = SimpleCNN(dropout=float(config["model"]["dropout"])).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"[INFO] Analizzando errori sul test set...")
    print(f"[INFO] Salverò fino a {max_to_save} immagini errate")

    # Traccia gli errori
    errors = []
    confusion_pairs = Counter()
    error_images = []  # Per la griglia visuale
    error_labels = []
    error_preds = []

    total_errors = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (x, y_bin, y_cifar) in enumerate(test_loader):
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu()
            
            total_samples += len(y_bin)

            for i in range(len(y_bin)):
                true_bin = int(y_bin[i])
                pred_bin = int(preds[i].item())

                if true_bin != pred_bin:
                    total_errors += 1
                    cif = int(y_cifar[i])
                    confusion_pairs[(true_bin, pred_bin, cif)] += 1

                    if len(errors) < max_to_save:
                        # Denormalizza l'immagine per visualizzazione corretta
                        img_normalized = x[i].cpu()
                        img_denormalized = denormalize(img_normalized)
                        
                        # Clamp i valori tra 0 e 1
                        img_denormalized = torch.clamp(img_denormalized, 0, 1)
                        
                        # Salva l'immagine denormalizzata
                        filename = f"err_{len(errors):04d}_true{true_bin}_pred{pred_bin}_cifar{cif}_{CIFAR10_LABELS[cif]}.png"
                        save_image(img_denormalized, os.path.join(out_dir, filename))
                        
                        # Salva per la griglia
                        error_images.append(img_denormalized)
                        error_labels.append(true_bin)
                        error_preds.append(pred_bin)
                        
                        errors.append({
                            "file": filename,
                            "true_bin": true_bin,
                            "true_label": "Animal" if true_bin == 0 else "Vehicle",
                            "pred_bin": pred_bin,
                            "pred_label": "Animal" if pred_bin == 0 else "Vehicle",
                            "cifar_label": cif,
                            "cifar_name": CIFAR10_LABELS[cif],
                        })

    # Crea la griglia visuale delle immagini errate
    if error_images:
        grid_path = os.path.join(out_dir, "error_grid.png")
        create_error_grid(error_images[:min(len(error_images), 32)], 
                         error_labels[:min(len(error_labels), 32)],
                         error_preds[:min(len(error_preds), 32)],
                         grid_path)

    # Aggrega i pattern di confusione
    top_patterns = confusion_pairs.most_common(20)
    summary = {
        "total_samples": total_samples,
        "total_errors": total_errors,
        "error_rate": total_errors / total_samples if total_samples > 0 else 0,
        "saved_misclassified": len(errors),
        "top_confusions": [
            {
                "true_bin": t,
                "true_label": "Animal" if t == 0 else "Vehicle",
                "pred_bin": p,
                "pred_label": "Animal" if p == 0 else "Vehicle",
                "cifar_label": c,
                "cifar_name": CIFAR10_LABELS[c],
                "count": n
            }
            for ((t, p, c), n) in top_patterns
        ]
    }

    # Salva i risultati
    with open(os.path.join(out_dir, "misclassified.json"), "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Stampa statistiche
    print(f"\n[INFO] Analisi completata!")
    print(f"[INFO] Totale campioni: {total_samples}")
    print(f"[INFO] Totale errori: {total_errors} ({summary['error_rate']*100:.2f}%)")
    print(f"[INFO] Immagini salvate: {len(errors)}")
    print(f"[INFO] File salvati in: {out_dir}")
    print(f"\n[INFO] Top 5 pattern di confusione:")
    for i, pattern in enumerate(summary["top_confusions"][:5], 1):
        print(f"  {i}. {pattern['cifar_name']} ({pattern['true_label']}) -> {pattern['pred_label']}: {pattern['count']} volte")


if __name__ == "__main__":
    main()