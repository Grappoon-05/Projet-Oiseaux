import argparse
from pathlib import Path
from ultralytics import YOLO

from select_device import device_config

# Extensions acceptées
EXTENSIONS_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODELE_PAR_DEFAUT = Path(__file__).resolve().parent / "yolo11n.pt"
DOSSIER_TEST_DEFAUT = Path("./dataset_split/test")
DOSSIER_SORTIE_DEFAUT = Path("./runs/yolo_test")


def options():
    parser = argparse.ArgumentParser(description="Test rapide du modèle YOLO sur des images.")
    parser.add_argument("source", type=Path, nargs="?", default=DOSSIER_TEST_DEFAUT, help="Image ou dossier à tester")
    parser.add_argument("--model", type=Path, default=MODELE_PAR_DEFAUT, help="Fichier .pt du modèle")
    parser.add_argument("--confidence", type=float, default=0.25, help="Seuil de confiance minimum")
    parser.add_argument("--output", type=Path, default=DOSSIER_SORTIE_DEFAUT, help="Dossier où enregistrer les images avec boîtes")
    return parser.parse_args()


# Récupère la liste des images à tester
def recuperer_images(source):
    if source.is_file():
        return [source]

    if not source.is_dir():
        raise FileNotFoundError(f"Dossier ou image introuvable : {source}")

    images = [
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS_IMAGES
    ]
    return sorted(images)


# Teste YOLO sur les images et affiche les détections
def tester_yolo(args):
    if not args.model.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {args.model}")

    images = recuperer_images(args.source)
    appareil_str = str(device_config())
    print(f"Chargement du modèle YOLO : {args.model}")
    modele = YOLO(str(args.model))

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Test lancé sur {len(images)} image(s)...\n")

    nb_trouves = 0
    # On teste sur un échantillon de 20 images pour que ce soit rapide
    echantillon = images[:20]

    for i, img_path in enumerate(echantillon, start=1):
        resultats = modele.predict(
            source=str(img_path),
            conf=args.confidence,
            classes=[14],  # classe 14 = oiseau
            device=appareil_str,
            verbose=False,
            save=False,
        )

        res = resultats[0]
        # On vérifie si une boîte a été trouvée
        if res.boxes is not None and len(res.boxes) > 0:
            nb_trouves += 1
            conf = float(res.boxes.conf.max())
            print(f"[{i:02d}] {img_path.name:30} -> OISEAU TROUVÉ (confiance: {conf:.2f})")
            # On enregistre l'image annotée avec le cadre vert
            chemin_sauvegarde = args.output / img_path.name
            res.save(filename=str(chemin_sauvegarde))
        else:
            print(f"[{i:02d}] {img_path.name:30} -> Rien détecté")

    print("\n" + "=" * 45)
    print("Test terminé !")
    print(f"Oiseaux trouvés : {nb_trouves}/{len(echantillon)}")
    print(f"Images avec cadres sauvegardées dans : {args.output.resolve()}")
    print("=" * 45)


if __name__ == "__main__":
    tester_yolo(options())
