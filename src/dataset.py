"""
Dataset utilities:
- Load CIFAR-10
- Convert 10-class labels into binary labels: Animal vs Vehicle
- Split training set into train/validation sets per best practices ML
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


# CIFAR-10 class indices:
# 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
# 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
VEHICLE_CLASSES = {0, 1, 8, 9}
ANIMAL_CLASSES = {2, 3, 4, 5, 6, 7}


@dataclass(frozen=True)
class BinaryLabelMap:
    """Binary mapping: 0=Animal, 1=Vehicle."""
    animal_label: int = 0
    vehicle_label: int = 1

    def to_binary(self, cifar_label: int) -> int:
        """Convert CIFAR-10 label into binary."""
        if cifar_label in ANIMAL_CLASSES:
            return self.animal_label
        if cifar_label in VEHICLE_CLASSES:
            return self.vehicle_label
        raise ValueError(f"Unexpected CIFAR label: {cifar_label}")


class CIFAR10AnimalVehicle(Dataset):
    """
    Wraps CIFAR-10 dataset and returns:
    - image tensor
    - binary label (0=Animal, 1=Vehicle)
    - original cifar label (for analysis)
    """

    def __init__(self, root: str, train: bool, transform=None, download: bool = True) -> None:
        self.base = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
        self.mapper = BinaryLabelMap()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, cifar_label = self.base[idx]
        y = self.mapper.to_binary(cifar_label)
        return img, y, cifar_label


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Data augmentation:
    - Train: random crop + flip (standard for CIFAR)
    - Test: only normalize
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, test_tf


def make_loaders(
    root_dir: str, 
    batch_size: int, 
    num_workers: int, 
    val_split_ratio: float = 0.2,
    seed: int = 42,
    device: str = "auto"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea train, validation e test dataloaders.
    
    IMPORTANTE: Il test set viene usato SOLO per la valutazione finale.
    Durante il training usiamo il validation set per monitorare le prestazioni
    e evitare overfitting.
    
    Args:
        root_dir: Directory dove salvare/scaricare CIFAR-10
        batch_size: Dimensione del batch per ogni iterazione
        num_workers: Numero di worker per il DataLoader (multiprocessing)
        val_split_ratio: Percentuale del training set da usare come validation (es. 0.2 = 20%)
        seed: Seed per la riproducibilità dello split
        device: Device da usare ("auto", "cpu", "cuda", "mps")
                Su MPS (Apple Silicon), num_workers viene automaticamente impostato a 0
        
    Returns:
        Tuple di (train_loader, val_loader, test_loader)
    """
    train_tf, test_tf = build_transforms()

    # Carica il dataset completo di training (50,000 immagini)
    full_train_ds = CIFAR10AnimalVehicle(root=root_dir, train=True, transform=train_tf, download=True)
    
    # Carica il test set (10,000 immagini) - NON viene toccato durante il training
    test_ds = CIFAR10AnimalVehicle(root=root_dir, train=False, transform=test_tf, download=True)

    # Dividi il training set in train e validation usando random_split
    # Questo mantiene la riproducibilità usando lo stesso seed
    total_train_size = len(full_train_ds)
    val_size = int(total_train_size * val_split_ratio)
    train_size = total_train_size - val_size
    
    # Usa seed per garantire riproducibilità dello split
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size], generator=generator)
    
    print(f"[INFO] Dataset split: Train={train_size}, Val={val_size}, Test={len(test_ds)}")

    # Su macOS con MPS, num_workers > 0 può causare problemi di condivisione memoria
    # Imposta automaticamente num_workers=0 se si usa MPS
    # torch è già importato all'inizio del file
    if device == "auto":
        if torch.backends.mps.is_available():
            device_type = "mps"
        elif torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"
    else:
        device_type = device
    
    # Su MPS, usa num_workers=0 e pin_memory=False per evitare problemi
    if device_type == "mps":
        effective_num_workers = 0
        pin_memory = False
        print("[INFO] MPS rilevato: usando num_workers=0 e pin_memory=False per compatibilità")
    else:
        effective_num_workers = num_workers
        pin_memory = True

    # Crea i DataLoader
    # Train: shuffle=True per mescolare i dati ad ogni epoca
    # Val/Test: shuffle=False perché non serve mescolare per la valutazione
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=effective_num_workers, 
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=effective_num_workers, 
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=effective_num_workers, 
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader