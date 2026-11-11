import sys
import csv
import json
from pathlib import Path
from PIL import Image
import torch
from torchvision import models
from ultralytics import YOLO

from select_device import device_config

# Inférence demandée en Annexe A du sujet :
# python predict.py <dossier d'images> <dossier de sortie>
# Sortie : un fichier predictions.csv avec colonnes exactes :
# image_id,sequence_id,label,confidence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_classes(classes_path="classes.json"):
    with open(classes_path, "r", encoding="utf-8") as f:
        raw = json.load(f)["classes"]
    classes = []
    for c in raw:
        if c not in ("autre_oiseau", "rien", "incertain"):
            classes.append(c.strip().replace(" ", "_"))
    return sorted(list(set(classes)))


def predict(input_folder, output_folder, weights_path="resnet18_birds.pth", yolo_model="yolo11n.pt"):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    device = device_config()
    classes = get_classes("classes.json")

    # Transformations officielles ImageNet ResNet18
    poids = models.ResNet18_Weights.DEFAULT
    transforms_resnet = poids.transforms()

    # Chargement du classifieur ResNet18
    modele = models.resnet18(weights=None)
    modele.fc = torch.nn.Linear(modele.fc.in_features, len(classes))

    # Recherche des poids disponibles
    poids_file = Path(weights_path)
    if not poids_file.exists():
        poids_alt = Path("output/model_weights.pth")
        if poids_alt.exists():
            poids_file = poids_alt
        else:
            poids_file = Path("baseline_cnn.pth")

    if poids_file.exists():
        modele.load_state_dict(torch.load(poids_file, map_location=device))
        print(f"Poids chargés depuis : {poids_file}")
    else:
        print("Attention : aucun fichier de poids trouvé, utilisation des poids aléatoires.")

    modele = modele.to(device)
    modele.eval()

    # Détecteur YOLO (page 5-6 du sujet) :
    # detecteur = YOLO("yolo11s.pt")
    # resultats = detecteur(chemin_image, classes=[14]) # 14 = bird
    detecteur_yolo = None
    if Path(yolo_model).exists():
        detecteur_yolo = YOLO(yolo_model)
        print(f"Détecteur YOLO chargé : {yolo_model}")

    # Récupération récursive de toutes les images
    images = [
        f for f in input_folder.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort()
    print(f"Nombre d'images à prédire : {len(images)}")

    csv_path = output_folder / "predictions.csv"

    # Écriture du CSV au format exact de l'Annexe A
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "sequence_id", "label", "confidence"])

        for idx, img_path in enumerate(images, start=1):
            image_id = img_path.stem  # nom du fichier sans extension ni chemin
            sequence_id = "0"         # identifiant de séquence par défaut

            img_pil = Image.open(img_path).convert("RGB")
            crop_a_classifier = img_pil
            a_un_oiseau = True
            confidence_yolo = 1.0

            # Détection YOLO si disponible
            if detecteur_yolo is not None:
                # Code exact du sujet :
                resultats = detecteur_yolo(str(img_path), classes=[14], verbose=False)
                boxes = resultats[0].boxes
                if boxes is not None and len(boxes) > 0:
                    # Extrait exact du sujet page 6 :
                    # x1, y1, x2, y2 = boite.xyxy[0].tolist()
                    # confiance = float(boite.conf[0])
                    idx_max = int(boxes.conf.argmax())
                    x1, y1, x2, y2 = boxes.xyxy[idx_max].tolist()
                    confidence_yolo = float(boxes.conf[idx_max])
                    crop_a_classifier = img_pil.crop((x1, y1, x2, y2))
                else:
                    a_un_oiseau = False

            # Sujet page 6 : absence de détection sur piège photo -> classe "rien"
            if not a_un_oiseau:
                label_final = "rien"
                conf_finale = 0.90
            else:
                # Classification ResNet18 sur le crop
                tensor = transforms_resnet(crop_a_classifier).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = modele(tensor)
                    probas = torch.softmax(logits, dim=1)
                    score_max, pred_idx = probas.max(dim=1)

                label_final = classes[pred_idx.item()]
                conf_finale = round(float(score_max.item()), 4)

            writer.writerow([image_id, sequence_id, label_final, conf_finale])

            if idx % 50 == 0 or idx == len(images):
                print(f"[{idx}/{len(images)}] {image_id} -> {label_final} ({conf_finale:.2f})")

    print(f"\nFichier CSV généré avec succès : {csv_path.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <dossier d'images> <dossier de sortie>")
        print("Exécution avec dossiers par défaut (dataset_split/test vers ./output)...")
        in_dir = "./dataset_split/test"
        out_dir = "./output"
    else:
        in_dir = sys.argv[1]
        out_dir = sys.argv[2]

    predict(in_dir, out_dir)
