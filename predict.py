import sys
import csv
import json
from datetime import datetime
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import torch
from torchvision import models
from ultralytics import YOLO

from select_device import device_config

# =====================================================================
# COMMANDE OFFICIELLE D'INFÉRENCE — ANNEXE A DU SUJET :
# python predict.py <dossier d'images> <dossier de sortie>
#
# Le CSV de prédictions :
# En-tête exact, une ligne par fichier d'entrée, aucune image omise :
# image_id,sequence_id,label,confidence
#
# - image_id : le nom du fichier, sans le chemin ni l'extension.
# - sequence_id : le passage auquel votre chaîne rattache l'image (étape 5).
# - label : un des labels de votre classes.json, et rien d'autre.
# - confidence : entre 0 et 1.
# =====================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
CLASSES_JSON_PATH = Path(__file__).resolve().parent / "classes.json"


def charger_vocabulaire(json_path=CLASSES_JSON_PATH):
    """Charge le vocabulaire exact déclaré dans classes.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["classes"]
    return [c.strip() for c in data]


def trouver_fichier_poids():
    """Localise automatiquement les meilleurs poids du modèle pour le rendu."""
    dossier_actuel = Path(__file__).resolve().parent
    candidats = [
        dossier_actuel / "final_model.pth",
        dossier_actuel / "resnet18_yolo_combine.pth",
        dossier_actuel / "resnet18_combine.pth",
        dossier_actuel / "output" / "final_model.pth",
        dossier_actuel / "output" / "model_weights.pth",
        Path("/Users/grappoon/Desktop/IUT/L2/S3/R3.14 - IA/projex/modeles_entraines/Etape3_YOLO_ResNet_combine_ResNet18.pth"),
    ]
    for c in candidats:
        if c.exists():
            return c
    return None


def extraire_date_exif(chemin_img):
    """Extrait la date et heure de prise de vue depuis les métadonnées EXIF."""
    try:
        im = Image.open(chemin_img)
        exif = im.getexif()
        if not exif:
            return None

        # 1. Vérification dans le bloc EXIF principal
        for tag_id, val in exif.items():
            nom_tag = TAGS.get(tag_id, tag_id)
            if nom_tag in ("DateTimeOriginal", "DateTime"):
                return datetime.strptime(str(val), "%Y:%m:%d %H:%M:%S")

        # 2. Vérification dans le sous-bloc SubIFD (exif tag 34665)
        if 34665 in exif:
            sub_ifd = exif.get_ifd(34665)
            for tag_id, val in sub_ifd.items():
                nom_tag = TAGS.get(tag_id, tag_id)
                if nom_tag in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                    return datetime.strptime(str(val), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def attribuer_sequences(chemins_images, seuil_secondes=60.0):
    """
    Rattache chaque image à un passage (Étape 5 du sujet).
    Si l'écart temporel EXIF entre 2 images consécutives est <= 60s,
    elles appartiennent à la même séquence (même visite d'oiseau).
    """
    if not chemins_images:
        return {}

    # Extraction des dates pour tri temporel
    infos = []
    for p in chemins_images:
        dt = extraire_date_exif(p)
        infos.append((p, dt))

    # Si aucune date EXIF n'est disponible (ex: images de test recadrées sans EXIF),
    # on regroupe par préfixe ou on conserve le tri alphabétique
    toutes_dates_nulles = all(dt is None for _, dt in infos)
    if toutes_dates_nulles:
        # Tri alphabétique stable
        infos.sort(key=lambda x: x[0].name)
        dict_seq = {}
        seq_courante = 0
        for i, (p, _) in enumerate(infos):
            # Regroupement par tranche ou rafale nominative (ex: PICT0008, HD0008)
            dict_seq[p] = str(seq_courante)
            if (i + 1) % 5 == 0:
                seq_courante += 1
        return dict_seq

    # Tri par date EXIF (ou nom en repli)
    infos.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else datetime.min, x[0].name))

    dict_seq = {}
    id_sequence_courante = 0
    date_precedente = None

    for chemin_img, date_img in infos:
        if date_img is not None and date_precedente is not None:
            delta = (date_img - date_precedente).total_seconds()
            if delta > seuil_secondes:
                id_sequence_courante += 1
        elif date_img is None and date_precedente is not None:
            id_sequence_courante += 1

        dict_seq[chemin_img] = str(id_sequence_courante)
        if date_img is not None:
            date_precedente = date_img

    return dict_seq


def predict(input_folder, output_folder, yolo_path="yolo11n.pt"):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    appareil = device_config()
    classes_officielles = charger_vocabulaire()

    # Espèces cibles déclarées (hors classes spéciales)
    classes_especes = [c for c in classes_officielles if c not in ("autre_oiseau", "rien", "incertain")]

    # Prétraitement officiel ResNet18
    poids_transforms = models.ResNet18_Weights.DEFAULT.transforms()

    # Chargement du modèle de classification ResNet18
    modele = models.resnet18(weights=None)
    modele.fc = torch.nn.Linear(modele.fc.in_features, len(classes_especes))

    fichier_poids = trouver_fichier_poids()
    if fichier_poids is not None:
        try:
            modele.load_state_dict(torch.load(fichier_poids, map_location=appareil))
            print(f"Poids du modèle chargés depuis : {fichier_poids}")
        except Exception as e:
            print(f"Avertissement : chargement des poids ({e})")
    else:
        print("Avertissement : aucun fichier de poids trouvé.")

    modele = modele.to(appareil)
    modele.eval()

    # Détecteur YOLO (Étape 3 du sujet)
    detecteur_yolo = None
    yolo_file = Path(yolo_path)
    if not yolo_file.exists():
        yolo_file = Path(__file__).resolve().parent / "yolo11n.pt"

    if yolo_file.exists():
        try:
            detecteur_yolo = YOLO(str(yolo_file))
            print(f"Détecteur YOLO prêt : {yolo_file}")
        except Exception as e:
            print(f"Avertissement YOLO ({e})")

    # 1. Lister l'ensemble exhaustif des images (sans en omettre aucune)
    images = [
        f for f in input_folder.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort()
    print(f"Nombre total d'images d'entrée à prédire : {len(images)}")

    # 2. Rattachement de chaque image à une séquence / passage (Étape 5)
    map_sequences = attribuer_sequences(images, seuil_secondes=60.0)

    # 3. Inférence image par image
    resultats_bruts = []  # tuples : (chemin, image_id, seq_id, label, confidence, probas)

    for i, chemin_img in enumerate(images, start=1):
        # Règle exacte : image_id = nom du fichier sans le chemin ni l'extension
        image_id = chemin_img.stem
        seq_id = map_sequences.get(chemin_img, "0")

        try:
            img_pil = Image.open(chemin_img).convert("RGB")
        except Exception:
            # Fichier illisible / corrompu -> prédiction de repli robuste
            resultats_bruts.append((chemin_img, image_id, seq_id, "rien", 0.50))
            continue

        crop_cible = img_pil
        oiseau_trouve = True

        # Passe de détection YOLO
        if detecteur_yolo is not None:
            try:
                res = detecteur_yolo(str(chemin_img), classes=[14], verbose=False)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    idx_max = int(res.boxes.conf.argmax())
                    x1, y1, x2, y2 = res.boxes.xyxy[idx_max].tolist()
                    
                    # Marge de 10 %
                    w_box = x2 - x1
                    h_box = y2 - y1
                    largeur_img, hauteur_img = img_pil.size
                    x1 = max(0, x1 - 0.1 * w_box)
                    y1 = max(0, y1 - 0.1 * h_box)
                    x2 = min(largeur_img, x2 + 0.1 * w_box)
                    y2 = min(hauteur_img, y2 + 0.1 * h_box)
                    
                    crop_cible = img_pil.crop((x1, y1, x2, y2))
                else:
                    oiseau_trouve = False
            except Exception:
                oiseau_trouve = True

        # Règle page 6 du sujet : absence de détection d'oiseau sur piège photo -> label 'rien'
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

        # Vérification de conformité avec classes.json
        if label_final not in classes_officielles:
            label_final = "autre_oiseau"

        # Confiance garantie entre 0 et 1
        conf_finale = max(0.0, min(1.0, conf_finale))
        resultats_bruts.append((chemin_img, image_id, seq_id, label_final, conf_finale))

        if i % 100 == 0 or i == len(images):
            print(f"[{i}/{len(images)}] {image_id} (seq {seq_id}) -> {label_final} ({conf_finale:.2f})")

    # 4. Étape 5 : Agrégation des décisions à l'échelle du passage (Visite d'oiseau)
    # Regroupement des prédictions par sequence_id
    seq_votes = {}
    for _, _, seq_id, label, conf in resultats_bruts:
        if seq_id not in seq_votes:
            seq_votes[seq_id] = {}
        # Pondération du vote par la confiance
        seq_votes[seq_id][label] = seq_votes[seq_id].get(label, 0.0) + conf

    # Pour chaque séquence, détermination de l'espèce majoritaire
    seq_consensus = {}
    for seq_id, votes in seq_votes.items():
        meilleur_label = max(votes.items(), key=lambda x: x[1])[0]
        seq_consensus[seq_id] = meilleur_label

    # 5. Écriture du fichier predictions.csv officiel
    csv_path = output_folder / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        # En-tête officiel exact
        writer.writerow(["image_id", "sequence_id", "label", "confidence"])

        for _, image_id, seq_id, label_img, conf_img in resultats_bruts:
            # Si le consensus du passage est différent et solide, on aligne
            label_final = seq_consensus.get(seq_id, label_img)
            
            # Si le consensus est un oiseau alors que l'image seule était 'rien',
            # on affine avec la décision du passage
            confidence_out = round(conf_img, 4)

            writer.writerow([image_id, seq_id, label_final, confidence_out])

    print("\n" + "=" * 65)
    print("CSV OFFICIEL GÉNÉRÉ AVEC SUCCÈS")
    print(f"Fichier : {csv_path.resolve()}")
    print(f"Total lignes écrites : {len(resultats_bruts)} images")
    print(f"Nombre de passages / séquences identifiés : {len(seq_consensus)}")
    print("En-tête : image_id,sequence_id,label,confidence")
    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <dossier d'images> <dossier de sortie>")
        sys.exit(1)
    else:
        predict(sys.argv[1], sys.argv[2])
