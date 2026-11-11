import sys
import csv
import json
from pathlib import Path
from PIL import Image
import torch
from torchvision import models
from ultralytics import YOLO

from select_device import device_config

EXTENSIONS_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def charger_classes(fichier_json="classes.json"):
    with open(fichier_json, "r", encoding="utf-8") as f:
        donnees = json.load(f)["classes"]
    classes = []
    for c in donnees:
        if c not in ("autre_oiseau", "rien", "incertain"):
            classes.append(c.strip().replace(" ", "_"))
    return sorted(list(set(classes)))


def predict(input_folder, output_folder, weights_path="resnet18_birds.pth", yolo_model="yolo11n.pt"):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    appareil = device_config()
    classes = charger_classes("classes.json")

    # Prétraitement ImageNet pour ResNet18
    poids = models.ResNet18_Weights.DEFAULT
    transforms_resnet = poids.transforms()

    # Reconstitution du modèle de classification
    modele = models.resnet18(weights=None)
    modele.fc = torch.nn.Linear(modele.fc.in_features, len(classes))

    # Chargement des poids
    chemin_poids = Path(weights_path)
    if not chemin_poids.exists():
        if Path("output/model_weights.pth").exists():
            chemin_poids = Path("output/model_weights.pth")
        else:
            chemin_poids = Path("baseline_cnn.pth")

    if chemin_poids.exists():
        modele.load_state_dict(torch.load(chemin_poids, map_location=appareil))
        print(f"Poids chargés depuis : {chemin_poids}")
    else:
        print("Aucun fichier .pth trouvé, prédictions avec poids non entraînés.")

    modele = modele.to(appareil)
    modele.eval()

    # Détecteur YOLO (classe 14 = bird)
    detecteur_yolo = None
    if Path(yolo_model).exists():
        detecteur_yolo = YOLO(yolo_model)
        print(f"Détecteur YOLO prêt : {yolo_model}")

    images = [
        f for f in input_folder.rglob("*")
        if f.is_file() and f.suffix.lower() in EXTENSIONS_IMAGES
    ]
    images.sort()
    print(f"Images à analyser : {len(images)}")

    csv_path = output_folder / "predictions.csv"

    # Écriture du CSV au format de l'Annexe A
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["image_id", "sequence_id", "label", "confidence"])

        for i, chemin_img in enumerate(images, start=1):
            image_id = chemin_img.stem
            sequence_id = "0"

            img_pil = Image.open(chemin_img).convert("RGB")
            crop_cible = img_pil
            oiseau_trouve = True

            # Détection YOLO
            if detecteur_yolo is not None:
                res = detecteur_yolo(str(chemin_img), classes=[14], verbose=False)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    idx_max = int(res.boxes.conf.argmax())
                    x1, y1, x2, y2 = res.boxes.xyxy[idx_max].tolist()
                    crop_cible = img_pil.crop((x1, y1, x2, y2))
                else:
                    oiseau_trouve = False

            # Gestion de l'absence d'oiseau
            if not oiseau_trouve:
                label_final = "rien"
                conf_finale = 0.95
            else:
                # Classification ResNet18 sur l'oiseau isolé
                tensor = transforms_resnet(crop_cible).unsqueeze(0).to(appareil)
                with torch.no_grad():
                    logits = modele(tensor)
                    probas = torch.softmax(logits, dim=1)
                    score_max, idx_classe = probas.max(dim=1)

                label_final = classes[idx_classe.item()]
                conf_finale = round(float(score_max.item()), 4)

            writer.writerow([image_id, sequence_id, label_final, conf_finale])

            if i % 50 == 0 or i == len(images):
                print(f"[{i}/{len(images)}] {image_id} -> {label_final} ({conf_finale:.2f})")

    print(f"\nFichier CSV généré dans : {csv_path.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <dossier d'images> <dossier de sortie>")
        print("Lancement par défaut : ./dataset_split/test vers ./output ...")
        predict("./dataset_split/test", "./output")
    else:
        predict(sys.argv[1], sys.argv[2])
