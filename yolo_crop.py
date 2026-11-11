import argparse
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from select_device import device_config

# Extensions d'images valides
EXTENSIONS_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DOSSIER_SOURCE_DEFAUT = Path("./dataset_split")
DOSSIER_SORTIE_DEFAUT = Path("./dataset_cropped")
MODELE_YOLO_DEFAUT = Path(__file__).resolve().parent / "yolo11n.pt"


def options():
    # Définition des paramètres avec valeurs par défaut
    parser = argparse.ArgumentParser(description="Recadrage des oiseaux dans les photos avec YOLO.")
    parser.add_argument("--source", type=Path, default=DOSSIER_SOURCE_DEFAUT, help="Dossier d'entrée (images)")
    parser.add_argument("--output", type=Path, default=DOSSIER_SORTIE_DEFAUT, help="Dossier de sortie (crops)")
    parser.add_argument("--model", type=Path, default=MODELE_YOLO_DEFAUT, help="Modèle YOLO")
    parser.add_argument("--confidence", type=float, default=0.25, help="Seuil de confiance minimum")
    parser.add_argument("--padding", type=float, default=0.10, help="Marge autour de l'oiseau (10%)")
    return parser.parse_args()


# Fonction qui découpe l'oiseau dans une image
def recadrer_oiseau(chemin_image, modele_yolo, appareil, seuil_confiance=0.25, marge=0.10):
    img = Image.open(chemin_image).convert("RGB")
    largeur, hauteur = img.size

    # On demande à YOLO de détecter seulement la classe 14 ("bird" dans COCO)
    resultats = modele_yolo.predict(
        source=str(chemin_image),
        conf=seuil_confiance,
        classes=[14],
        device=appareil,
        verbose=False,
        save=False,
    )

    # Si rien n'est trouvé, on renvoie l'image d'origine
    if not resultats or resultats[0].boxes is None or len(resultats[0].boxes) == 0:
        return img, False, 0.0

    boites = resultats[0].boxes
    # On choisit l'oiseau avec la meilleure confiance
    idx_max = int(boites.conf.argmax())
    confiance_max = float(boites.conf[idx_max])
    x1, y1, x2, y2 = boites.xyxy[idx_max].tolist()

    # On ajoute une petite marge de 10% pour ne pas couper le bec ou les plumes
    largeur_boite = x2 - x1
    hauteur_boite = y2 - y1
    marge_x = largeur_boite * marge
    marge_y = hauteur_boite * marge

    # On s'assure de ne pas sortir des limites de la photo
    crop_x1 = max(0, int(x1 - marge_x))
    crop_y1 = max(0, int(y1 - marge_y))
    crop_x2 = min(largeur, int(x2 + marge_x))
    crop_y2 = min(hauteur, int(y2 + marge_y))

    # On découpe
    image_recadree = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    return image_recadree, True, confiance_max


def executer_recadrage(args):
    if not args.model.is_file():
        raise FileNotFoundError(f"Modèle YOLO introuvable : {args.model}")

    appareil_str = str(device_config())
    print(f"Chargement de YOLO : {args.model}")
    modele = YOLO(str(args.model))

    dossier_in = args.source
    dossier_out = args.output
    dossier_out.mkdir(parents=True, exist_ok=True)

    # On récupère toutes les images du dossier d'entrée
    toutes_images = [
        p for p in dossier_in.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS_IMAGES
    ]

    nb_total = len(toutes_images)
    print(f"Dossier source : {dossier_in}")
    print(f"Dossier de sortie : {dossier_out}")
    print(f"Nombre d'images à traiter : {nb_total}\n")

    nb_detectes = 0
    nb_sans_detection = 0

    for i, chemin_img in enumerate(toutes_images, start=1):
        # On garde la même arborescence (ex: train/Parus_major/img.jpg)
        chemin_relatif = chemin_img.relative_to(dossier_in)
        chemin_dest = dossier_out / chemin_relatif
        chemin_dest.parent.mkdir(parents=True, exist_ok=True)

        img_crop, detecte, conf = recadrer_oiseau(
            chemin_img, modele, appareil=appareil_str, seuil_confiance=args.confidence, marge=args.padding
        )

        if detecte:
            nb_detectes += 1
        else:
            nb_sans_detection += 1

        img_crop.save(chemin_dest)

        if i % 100 == 0 or i == nb_total:
            pct_detect = (nb_detectes / i) * 100
            print(f"Avancement : [{i}/{nb_total}] - Détectés : {nb_detectes} ({pct_detect:.1f}%) - Non trouvés : {nb_sans_detection}")

    print("\n" + "=" * 45)
    print("Recadrage terminé !")
    print(f"Images avec détection d'oiseau : {nb_detectes}/{nb_total} ({nb_detectes/nb_total*100:.1f}%)")
    print(f"Images conservées telles quelles : {nb_sans_detection}")
    print(f"Dossier créé : {dossier_out.resolve()}")
    print("=" * 45)


if __name__ == "__main__":
    executer_recadrage(options())
