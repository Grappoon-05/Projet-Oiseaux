"""Teste le modèle YOLO (yolo11n.pt) sur une image ou un dossier d'images.
Adapté de test_YOLO.py du projet Soteria.

Exemples :
    python3 test_yolo.py
    python3 test_yolo.py mon_image.jpg
    python3 test_yolo.py dataset_split/test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from select_device import device_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODEL = Path(__file__).resolve().parent / "yolo11n.pt"
DEFAULT_TEST_DIR = Path("./dataset_split/test")
DEFAULT_OUTPUT = Path("./runs/yolo_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teste le modèle YOLO sur une image ou un dossier d'images."
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=DEFAULT_TEST_DIR,
        help="Image ou dossier à tester (défaut: ./dataset_split/test).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Chemin du modèle YOLO (défaut: ./yolo11n.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Seuil de confiance (défaut: 0.25).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Dossier de sortie des images annotées (défaut: ./runs/yolo_test).",
    )
    return parser.parse_args()


def get_image_list(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Format non supporté : {source.suffix}")
        return [source]

    if not source.is_dir():
        raise FileNotFoundError(f"Source introuvable : {source}")

    images = [
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise FileNotFoundError(f"Aucune image trouvée dans : {source}")
    return sorted(images)


def test_yolo(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(f"Modèle YOLO introuvable : {args.model}")

    sources = get_image_list(args.source)
    device = str(device_config())
    print(f"Chargement YOLO : {args.model} sur {device}")
    model = YOLO(str(args.model))

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Test sur {len(sources)} image(s) depuis {args.source}...\n")

    nb_detected = 0

    for idx, img_path in enumerate(sources[:30], start=1):  # Échantillon de test
        results = model.predict(
            source=str(img_path),
            conf=args.confidence,
            classes=[14],  # 14 = bird
            device=device,
            verbose=False,
            save=False,
        )

        res = results[0]
        has_bird = res.boxes is not None and len(res.boxes) > 0
        if has_bird:
            nb_detected += 1
            conf = float(res.boxes.conf.max())
            print(f"[{idx:02d}] {img_path.name:30} -> OISEAU DÉTECTÉ (conf: {conf:.2f})")
            # Sauvegarder l'image annotée
            out_img = args.output / img_path.name
            res.save(filename=str(out_img))
        else:
            print(f"[{idx:02d}] {img_path.name:30} -> Aucun oiseau")

    print("\n" + "=" * 50)
    print(f"Test terminé sur l'échantillon.")
    print(f"Oiseaux détectés : {nb_detected}/{min(30, len(sources))}")
    print(f"Images annotées sauvegardées dans : {args.output.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    test_yolo(parse_args())
