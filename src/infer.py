"""
Inference script:
- Load checkpoint
- Predict on a single image path OR on webcam frames (OpenCV)
This is the bridge to real-time monitoring cameras.
"""

from __future__ import annotations
import argparse
import os
import sys
import cv2
import torch
from torchvision import transforms
from PIL import Image

# Aggiungi il path del progetto a sys.path per permettere gli import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.cnn import SimpleCNN
from src.utils import load_config, pick_device

LABELS = {0: "Animal", 1: "Vehicle"}


def build_infer_transform() -> transforms.Compose:
    """Must match test normalization used during training."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_model(ckpt_path: str, device: torch.device, dropout: float) -> SimpleCNN:
    """Load model weights from checkpoint."""
    model = SimpleCNN(dropout=dropout).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_image(model: SimpleCNN, device: torch.device, image_path: str) -> int:
    """Predict binary class for a single image file."""
    tf = build_infer_transform()
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    pred = int(torch.argmax(logits, dim=1).item())
    return pred


def webcam_loop(model: SimpleCNN, device: torch.device) -> None:
    """
    Basic webcam loop:
    - Reads frames
    - Resizes to 32x32 and predicts
    - Shows label overlay
    NOTE: This is a demo. For production you would run detection + tracking + ROI crops.
    """
    cap = cv2.VideoCapture(0)
    tf = build_infer_transform()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((32, 32))
        x = tf(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())

        label = LABELS[pred]
        cv2.putText(frame, f"Pred: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        cv2.imshow("VisionTech - Animal vs Vehicle (demo)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to an image to classify")
    parser.add_argument("--webcam", action="store_true", help="Run webcam demo")
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    device = pick_device(config["train"]["device"])
    model = load_model(config["outputs"]["checkpoint_path"], device, dropout=float(config["model"]["dropout"]))

    if args.image:
        pred = predict_image(model, device, args.image)
        print(f"Prediction: {LABELS[pred]}")
    elif args.webcam:
        webcam_loop(model, device)
    else:
        print("Provide --image path OR --webcam")


if __name__ == "__main__":
    main()