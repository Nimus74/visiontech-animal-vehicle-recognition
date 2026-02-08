"""
CNN model for binary image classification: Animal vs Vehicle (CIFAR-10).

Questo modulo contiene l'architettura della rete neurale convoluzionale (CNN)
progettata specificamente per classificare immagini CIFAR-10 (32x32 pixel RGB)
in due classi: Animal (0) e Vehicle (1).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    CNN compatta adatta per classificazione binaria su CIFAR-10.
    
    L'architettura è composta da:
    1. Feature Extractor: 3 blocchi convoluzionali che estraggono features
    2. Classifier: MLP (Multi-Layer Perceptron) che classifica le features
    
    Args:
        dropout: Tasso di dropout da applicare nel classifier (default: 0.25)
                 Il dropout aiuta a prevenire overfitting disattivando casualmente
                 alcuni neuroni durante il training.
    
    Output:
        Tensor di forma [batch_size, 2] con logits per Animal e Vehicle
    """

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()

        # Feature Extractor: estrae caratteristiche dalle immagini
        # Ogni blocco riduce la dimensione spaziale e aumenta i canali
        self.features = nn.Sequential(
            # Blocco 1: Primo livello di features base
            # Input: [B, 3, 32, 32] -> Output: [B, 32, 16, 16]
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Convoluzione: 3 canali RGB -> 32 feature maps
            nn.BatchNorm2d(32),  # Normalizzazione per stabilizzare il training
            nn.ReLU(inplace=True),  # Attivazione non-lineare
            nn.MaxPool2d(2),  # Pooling: riduce dimensione da 32x32 a 16x16

            # Blocco 2: Features intermedie
            # Input: [B, 32, 16, 16] -> Output: [B, 64, 8, 8]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32 -> 64 feature maps
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Riduce da 16x16 a 8x8

            # Blocco 3: Features avanzate
            # Input: [B, 64, 8, 8] -> Output: [B, 128, 4, 4]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64 -> 128 feature maps
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Riduce da 8x8 a 4x4
        )

        # Classifier: MLP che classifica le features estratte
        # Input: [B, 128*4*4] = [B, 2048] -> Output: [B, 2]
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Appiattisce da [B, 128, 4, 4] a [B, 2048]
            nn.Linear(128 * 4 * 4, 256),  # Primo layer fully-connected: 2048 -> 256 neuroni
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # Dropout per prevenire overfitting
            nn.Linear(256, 2),  # Output layer: 256 -> 2 logits (Animal, Vehicle)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: calcola le predizioni del modello.
        
        Args:
            x: Tensor di input con forma [batch_size, 3, 32, 32]
               Rappresenta un batch di immagini RGB 32x32
        
        Returns:
            Tensor con forma [batch_size, 2] contenente i logits per ogni classe
            - logits[:, 0]: probabilità logaritmica per Animal
            - logits[:, 1]: probabilità logaritmica per Vehicle
            
        Nota: I logits devono essere passati attraverso softmax per ottenere probabilità
        o attraverso argmax per ottenere la classe predetta.
        """
        # Estrai features dalle immagini
        x = self.features(x)
        
        # Classifica le features
        x = self.classifier(x)
        
        return x