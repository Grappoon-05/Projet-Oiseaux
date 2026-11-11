"""Détection et recadrage d'oiseaux avec YOLO pour le projet Oiseaux.

Ce script adapte la logique de test_YOLO.py du projet Soteria :
- Utilise YOLO (yolo11n.pt) pour détecter la classe 14 ("bird" du jeu COCO).
- Isole la boîte englobante de l'oiseau et recadre l'image (avec marge de sécurité).
- Sauvegarde le crop pour alimenter le classifieur (Étape 3 du sujet).
- Si aucun oiseau n'est détecté :
    * Sur Data_Hackaton : conserve l'image complète (fallback pour ne pas perdre la donnée).
    * Sur le piège photo : classe directement en "rien" comme demandé par le sujet.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from PIL import Image
from ultralytics import YOLO

from select_device import device_config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_YOLO_MODEL = Path(__file__).resolve().parent / "yolo11n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recadre les oiseaux dans les images à l'aide d'un modèle YOLO."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("./dataset_split"),
        help="Dossier source contenant les images ou sous-dossiers (ex: dataset_split).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./dataset_cropped"),
        help="Dossier de sortie pour stocker les images recadrées.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_YOLO_MODEL,
        help="Chemin du modèle YOLO (défaut: yolo11n.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Seuil de confiance pour la détection d'oiseau (défaut: 0.25).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.10,
        help="Marge relative ajoutée autour de la boîte englobante (défaut: 10%%).",
    )
    return parser.parse_args()


def crop_bird_from_image(
    image_path: Path,
    model: YOLO,
    device: str,
    confidence: float = 0.25,
    padding: float = 0.10,
) -> tuple[Image.Image, bool, float]:
    """Détecte l'oiseau le plus confiant et retourne le crop (ou l'image entière si non trouvé).

    Returns:
        tuple: (PIL.Image recadrée ou originale, bool détection réussie, float confiance max).
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # Inférence YOLO en ciblant la classe 14 = "bird" de COCO
    results = model.predict(
        source=str(image_path),
        conf=confidence,
        classes=[14],  # 14 = bird
        device=device,
        verbose=False,
        save=False,
    )

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        # Aucun oiseau détecté avec le seuil demandé
        return img, False, 0.0

    boxes = results[0].boxes
    # On sélectionne la boîte avec la meilleure confiance
    best_idx = int(boxes.conf.argmax())
    best_conf = float(boxes.conf[best_idx])
    x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()

    # Ajout d'une marge (padding) pour ne pas couper le bout des plumes/bec
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

    # Récupération récursive de toutes les images
    all_images = [
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]

    total_images = len(all_images)
    print(f"Total d'images à traiter : {total_images}")

    detected_count = 0
    fallback_count = 0

    for idx, img_path in enumerate(all_images, start=1):
        # Préserve la structure de sous-dossiers (ex: dataset_split/train/Parus_major/...)
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
                f"Sans détection : {fallback_count}"
            )

    print("\n" + "=" * 55)
    print("Recadrage terminé avec succès !")
    print(f"Dossier de sortie : {output_dir.resolve()}")
    print(f"Images avec détection d'oiseau : {detected_count}/{total_images} ({detected_count/total_images*100:.1f}%)")
    print(f"Images sans détection (fallback conservé) : {fallback_count}")
    print("=" * 55)


if __name__ == "__main__":
    args = parse_args()
    process_dataset(args)
