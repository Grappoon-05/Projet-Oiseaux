import sys
import csv
import json
from pathlib import Path
from PIL import Image
import torch
from torchvision import models
from ultralytics import YOLO

from select_device import device_config

# Commande officielle d'inférence (Annexe A du sujet) :
# python predict.py <dossier d'images> <dossier de sortie>
#
# Le sujet exige :
# "Votre modèle final : les poids dans le dépôt, avec le classes.json correspondant.
#  Votre script d'inférence doit savoir où les trouver tout seul."

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHEMIN_POIDS_FINAL = Path(__file__).resolve().parent / "final_model.pth"
CLASSES_JSON_PATH = Path(__file__).resolve().parent / "classes.json"


def charger_vocabulaire(json_path=CLASSES_JSON_PATH):
    """Charge le vocabulaire exact déclaré dans classes.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["classes"]
    return [c.strip() for c in data]


def trouver_fichier_poids():
    """Localise automatiquement les meilleurs poids du modèle pour le rendu."""
    # 1. Poids final officiel du rendu à la racine du dépôt
    if CHEMIN_POIDS_FINAL.exists():
        return CHEMIN_POIDS_FINAL

    # 2. Modèles entraînés dans projex
    projex_dir = Path("/Users/grappoon/Desktop/IUT/L2/S3/R3.14 - IA/projex/modeles_entraines")
    candidats_projex = [
        projex_dir / "Etape3_YOLO_ResNet_combine_ResNet18.pth",
        projex_dir / "Etape2_ResNet18_combine_ResNet18.pth",
        projex_dir / "Etape2_ResNet18_laurens_ResNet18.pth",
    ]
    for c in candidats_projex:
        if c.exists():
            return c

    # 3. Poids locaux alternatifs
    for nom in ["resnet18_birds.pth", "output/model_weights.pth", "baseline_cnn.pth"]:
        p = Path(nom)
        if p.exists():
            return p

    return None


def predict(input_folder, output_folder, yolo_path="yolo11n.pt"):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    appareil = device_config()
    classes_officielles = charger_vocabulaire()

    # Classes d'espèces réelles (hors 'autre_oiseau' et 'rien')
    classes_especes = [c for c in classes_officielles if c not in ("autre_oiseau", "rien", "incertain")]

    # Prétraitement ImageNet officiel pour ResNet18
    poids_transforms = models.ResNet18_Weights.DEFAULT.transforms()

    # Reconstruction de ResNet18
    modele = models.resnet18(weights=None)
    modele.fc = torch.nn.Linear(modele.fc.in_features, len(classes_especes))

    fichier_poids = trouver_fichier_poids()
    if fichier_poids is not None:
        try:
            modele.load_state_dict(torch.load(fichier_poids, map_location=appareil))
            print(f"Modèle final chargé avec succès depuis : {fichier_poids}")
        except Exception as e:
            print(f"Erreur au chargement des poids ({e}), utilisation en mode non-entraîné.")
    else:
        print("Aucun fichier de poids trouvé. Placez final_model.pth à la racine.")

    modele = modele.to(appareil)
    modele.eval()

    # Détecteur YOLO (Étape 3)
    detecteur_yolo = None
    yolo_file = Path(yolo_path)
    if not yolo_file.exists():
        yolo_file = Path(__file__).resolve().parent / "yolo11n.pt"

    if yolo_file.exists():
        try:
            detecteur_yolo = YOLO(str(yolo_file))
            print(f"Détecteur YOLO prêt : {yolo_file}")
        except Exception as e:
            print(f"Avertissement YOLO ({e}), classification directe.")

    # Récupération des images à prédire
    images = [
        f for f in input_folder.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort()
    print(f"Nombre d'images à prédire : {len(images)}")

    csv_path = output_folder / "predictions.csv"

    # En-tête officiel exact : image_id,sequence_id,label,confidence
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["image_id", "sequence_id", "label", "confidence"])

        for i, chemin_img in enumerate(images, start=1):
            image_id = chemin_img.stem  # nom du fichier sans extension
            sequence_id = "0"          # identifiant de passage (Étape 5)

            try:
                img_pil = Image.open(chemin_img).convert("RGB")
            except Exception:
                # Fichier corrompu -> prédiction de repli
                writer.writerow([image_id, sequence_id, "rien", 0.50])
                continue

            crop_cible = img_pil
            oiseau_trouve = True

            # Passe YOLO si actif
            if detecteur_yolo is not None:
                res = detecteur_yolo(str(chemin_img), classes=[14], verbose=False)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    idx_max = int(res.boxes.conf.argmax())
                    x1, y1, x2, y2 = res.boxes.xyxy[idx_max].tolist()
                    crop_cible = img_pil.crop((x1, y1, x2, y2))
                else:
                    oiseau_trouve = False

            # Sujet page 6 : absence de détection sur piège photo -> "rien"
            if not oiseau_trouve:
                label_final = "rien"
                conf_finale = 0.95
            else:
                tensor = poids_transforms(crop_cible).unsqueeze(0).to(appareil)
                with torch.no_grad():
                    logits = modele(tensor)
                    probas = torch.softmax(logits, dim=1)
                    score_max, idx_classe = probas.max(dim=1)

                label_final = classes_especes[idx_classe.item()]
                conf_finale = round(float(score_max.item()), 4)

            # Vérification stricte que le label appartient bien à classes.json
            if label_final not in classes_officielles:
                label_final = "autre_oiseau"

            writer.writerow([image_id, sequence_id, label_final, conf_finale])

            if i % 100 == 0 or i == len(images):
                print(f"[{i}/{len(images)}] {image_id} -> {label_final} ({conf_finale:.2f})")

    print(f"\nFichier CSV généré avec succès dans : {csv_path.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <dossier d'images> <dossier de sortie>")
        print("Lancement par défaut : ./dataset_split/test vers ./output ...")
        predict("./dataset_split/test", "./output")
    else:
        predict(sys.argv[1], sys.argv[2])
