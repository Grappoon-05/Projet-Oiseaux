"""Détection et recadrage d'oiseaux avec YOLO (Étape 3 du sujet).

Lancement simple :
    python3 yolo_crop.py

Le script utilise automatiquement les paramètres par défaut :
- Source : ./dataset_split
- Sortie : ./dataset_cropped
- Modèle : ./yolo11n.pt
- Confiance : 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from select_device import device_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SOURCE = Path("./dataset_split")
DEFAULT_OUTPUT = Path("./dataset_cropped")
DEFAULT_MODEL = Path(__file__).resolve().parent / "yolo11n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recadre les oiseaux dans les images avec YOLO."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Dossier source des images (défaut: ./dataset_split).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Dossier de sortie des crops (défaut: ./dataset_cropped).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Modèle YOLO (défaut: ./yolo11n.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Seuil de confiance YOLO (défaut: 0.25).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.10,
        help="Marge relative autour de l'oiseau (défaut: 10%%).",
    )
    return parser.parse_args()


def crop_bird_from_image(
    image_path: Path,
    model: YOLO,
    device: str,
    confidence: float = 0.25,
    padding: float = 0.10,
) -> tuple[Image.Image, bool, float]:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    results = model.predict(
        source=str(image_path),
        conf=confidence,
        classes=[14],  # 14 = "bird" dans COCO
        device=device,
        verbose=False,
        save=False,
    )

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return img, False, 0.0

    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    best_conf = float(boxes.conf[best_idx])
    x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()

    w_box = x2 - x1
    h_box = y2 - y1
    pad_w = w_box * padding
    pad_h = h_box * padding

    crop_x1 = max(0, int(x1 - pad_w))
    crop_y1 = max(0, int(y1 - pad_h))
    crop_x2 = min(width, int(x2 + pad_w))
    crop_y2 = min(height, int(y2 + pad_h))

    cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    return cropped_img, True, best_conf


def process_dataset(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Modèle YOLO introuvable : {args.model}. "
            "Vérifiez que yolo11n.pt est bien présent à la racine."
        )

    device = str(device_config())
    print(f"Chargement du modèle YOLO : {args.model}")
    model = YOLO(str(args.model))

    source_dir = args.source
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = [
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]

    total_images = len(all_images)
    print(f"Source : {source_dir.resolve()}")
    print(f"Destination : {output_dir.resolve()}")
    print(f"Total d'images à traiter : {total_images}\n")

    detected_count = 0
    fallback_count = 0

    for idx, img_path in enumerate(all_images, start=1):
        rel_path = img_path.relative_to(source_dir)
        dest_path = output_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        cropped_img, detected, conf = crop_bird_from_image(
            img_path, model, device=device, confidence=args.confidence, padding=args.padding
        )

        if detected:
            detected_count += 1
        else:
            fallback_count += 1

        cropped_img.save(dest_path)

        if idx % 100 == 0 or idx == total_images:
            print(
                f"Progression : [{idx}/{total_images}] | "
                f"Détectés : {detected_count} ({detected_count/idx*100:.1f}%) | "
                f"Sans détection (fallback) : {fallback_count}"
            )

    print("\n" + "=" * 55)
    print("Recadrage YOLO terminé avec succès !")
    print(f"Dossier de sortie : {output_dir.resolve()}")
    print(f"Images avec oiseau détecté : {detected_count}/{total_images} ({detected_count/total_images*100:.1f}%)")
    print(f"Images sans détection : {fallback_count}")
    print("=" * 55)


def main() -> None:
    args = parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()
