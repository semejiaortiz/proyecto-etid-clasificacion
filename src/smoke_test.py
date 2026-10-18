#!/usr/bin/env python3
"""Prueba mínima del entorno, los datos y el checkpoint final de ETID."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0


CLASSES = ["Battery", "Cable", "Keyboard", "Laptop", "Mobile", "Mouse", "PCB"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Letterbox:
    def __init__(self, size: int = 224, fill=(0, 0, 0)) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = min(self.size / width, self.size / height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - resized.width) // 2, (self.size - resized.height) // 2))
        return canvas


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ["ETID_DATA_ROOT"]) if "ETID_DATA_ROOT" in os.environ else None,
        help="Raíz con las siete carpetas de clase de ETID (o ETID_DATA_ROOT).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_root is None:
        raise SystemExit("Defina --data-root o la variable ETID_DATA_ROOT.")

    repo = Path(__file__).resolve().parents[1]
    data_root = args.data_root.expanduser().resolve()
    checkpoint_path = repo / "models" / "efficientnet_b0_etid_selected.pt"
    manifest_path = repo / "models" / "model_manifest.json"
    split_path = repo / "data" / "splits" / "test.csv"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_checkpoint_sha = manifest["sha256"]
    actual_checkpoint_sha = sha256_file(checkpoint_path)
    if actual_checkpoint_sha != expected_checkpoint_sha:
        raise RuntimeError("El SHA-256 del checkpoint no coincide con model_manifest.json.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("classes") != CLASSES:
        raise RuntimeError("El orden de clases del checkpoint no coincide con el proyecto.")

    dropout = float(checkpoint["config"]["dropout"])
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, len(CLASSES)))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    row = pd.read_csv(split_path).iloc[0]
    image_path = data_root / Path(str(row["ruta"]))
    if not image_path.is_file():
        raise FileNotFoundError(f"No se encontró la imagen de prueba: {image_path}")
    if sha256_file(image_path) != row["sha256"]:
        raise RuntimeError("La imagen de prueba no coincide con su manifiesto.")

    transform = transforms.Compose(
        [Letterbox(224), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    with Image.open(image_path) as image, torch.inference_mode():
        logits = model(transform(image).unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)[0]

    if logits.shape != (1, len(CLASSES)) or not torch.isfinite(logits).all():
        raise RuntimeError("La inferencia produjo una salida inválida.")

    predicted_index = int(probabilities.argmax())
    result = {
        "resultado": "APROBADO",
        "imagen": str(row["ruta"]),
        "etiqueta_real": str(row["etiqueta_final"]),
        "prediccion": CLASSES[predicted_index],
        "confianza": round(float(probabilities[predicted_index]), 6),
        "salida": list(logits.shape),
        "checkpoint_sha256": actual_checkpoint_sha,
        "torch": torch.__version__,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
