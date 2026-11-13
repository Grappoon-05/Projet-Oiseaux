import sys
import json
import time
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from ultralytics import YOLO

from select_device import device_config

# =====================================================================
# COMMANDE OFFICIELLE D'ENTRAÎNEMENT — ANNEXE A DU SUJET :
# python train.py <racine des données> <dossier de sortie>
#
# Comportement :
# - Aucune interaction bloquante (ni input(), ni invite interactive).
# - Recherche automatique et flexible des images (racine directe, train/, ou sous-dossiers).
# - Intégration du pipeline complet de détection / recadrage YOLO (Étape 3).
# - Sauvegarde des poids dans <dossier de sortie>/final_model.pth (et model_weights.pth).
# - Sauvegarde de la courbe d'apprentissage loss dans <dossier de sortie>/loss_train.png.
# - Sauvegarde d'une copie de classes.json dans le dossier de sortie pour autonomie complète.
# =====================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
CLASSES_JSON_PATH = Path(__file__).resolve().parent / "classes.json"


def charger_vocabulaire(json_path=CLASSES_JSON_PATH):
    """Charge le vocabulaire exact déclaré dans classes.json."""
    if not Path(json_path).exists():
        json_path = Path("classes.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["classes"]
    classes = [c.strip() for c in data if c not in ("autre_oiseau", "rien", "incertain")]
    return sorted(list(set(classes)))


class BirdDataset(Dataset):
    """Dataset pour charger et optionnellement recadrer les images par classe exclusivement dans data_root."""
    def __init__(self, root, classes, transform=None, detecteur_yolo=None):
        self.root = Path(root).resolve()
        self.transform = transform
        self.detecteur_yolo = detecteur_yolo
        self.paths = []
        self.labels = []
        classe_vers_index = {nom: i for i, nom in enumerate(classes)}

        if not self.root.exists() or not self.root.is_dir():
            raise FileNotFoundError(f"Le dossier spécifié n'existe pas : {self.root}")

        # Parcours STRICTEMENT circonscrit à l'intérieur de self.root
        for nom in classes:
            # 1. Sous-dossier direct portant le nom de la classe dans self.root
            dossier_classe = self.root / nom
            if dossier_classe.is_dir():
                for fichier in sorted(dossier_classe.iterdir()):
                    if fichier.is_file() and fichier.suffix in IMAGE_EXTENSIONS:
                        self.paths.append(fichier)
                        self.labels.append(classe_vers_index[nom])
            else:
                # 2. Sous-dossiers situés strictement à l'intérieur de self.root (sans jamais en sortir)
                for sous_dossier in self.root.rglob(nom):
                    if sous_dossier.is_dir():
                        for fichier in sorted(sous_dossier.iterdir()):
                            if fichier.is_file() and fichier.suffix in IMAGE_EXTENSIONS:
                                self.paths.append(fichier)
                                self.labels.append(classe_vers_index[nom])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        chemin = self.paths[index]
        image = Image.open(chemin).convert("RGB")

        # Recadrage YOLO si actif (Étape 3 du sujet)
        if self.detecteur_yolo is not None:
            try:
                res = self.detecteur_yolo(str(chemin), classes=[14], verbose=False)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    idx_max = int(res.boxes.conf.argmax())
                    x1, y1, x2, y2 = res.boxes.xyxy[idx_max].tolist()
                    w_box = x2 - x1
                    h_box = y2 - y1
                    largeur, hauteur = image.size
                    x1 = max(0, x1 - 0.1 * w_box)
                    y1 = max(0, y1 - 0.1 * h_box)
                    x2 = min(largeur, x2 + 0.1 * w_box)
                    y2 = min(hauteur, y2 + 0.1 * h_box)
                    image = image.crop((x1, y1, x2, y2))
            except Exception:
                pass

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        return image, label


def train(data_root, output_dir, epochs=10, batch_size=32, lr=0.0005):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("COMMANDE OFFICIELLE D'ENTRAÎNEMENT (ANNEXE A)")
    print(f" - Racine des données  : {data_root.resolve()}")
    print(f" - Dossier de sortie   : {output_dir.resolve()}")
    print("=" * 65)

    appareil = device_config()
    classes = charger_vocabulaire()
    print(f"Nombre d'espèces cibles : {len(classes)}")

    # Initialisation de YOLO pour le recadrage (Étape 3)
    detecteur_yolo = None
    yolo_file = Path(__file__).resolve().parent / "yolo11n.pt"
    if yolo_file.exists():
        try:
            detecteur_yolo = YOLO(str(yolo_file))
            print(f"Détecteur YOLO prêt pour l'entraînement : {yolo_file.name}")
        except Exception as e:
            print(f"YOLO non activé ({e}), entraînement direct.")

    # Transformations officielles pour ResNet18
    poids_preentraines = models.ResNet18_Weights.DEFAULT
    transformations = poids_preentraines.transforms()

    # Préparation du modèle ResNet18 (Transfer Learning)
    modele = models.resnet18(weights=poids_preentraines)
    nb_features = modele.fc.in_features
    modele.fc = nn.Linear(nb_features, len(classes))
    modele = modele.to(appareil)

    dataset = BirdDataset(data_root, classes, transform=transformations, detecteur_yolo=detecteur_yolo)
    print(f"Images d'entraînement chargées : {len(dataset)}")

    if len(dataset) == 0:
        print(f"Erreur : aucune image trouvée dans {data_root}")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modele.parameters(), lr=lr)

    historique_perte = []
    print(f"\n--- Début de l'entraînement ({epochs} époques) ---")

    modele.train()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perte_cumulee = 0.0

        for images, cibles in loader:
            images = images.to(appareil)
            cibles = cibles.to(appareil)

            optimizer.zero_grad()
            sorties = modele(images)
            perte = criterion(sorties, cibles)
            perte.backward()
            optimizer.step()

            perte_cumulee += perte.item() * images.size(0)

        perte_moyenne = perte_cumulee / len(dataset)
        historique_perte.append(perte_moyenne)
        dt = time.time() - t0
        print(f"Époque [{epoch:02d}/{epochs:02d}] - Perte: {perte_moyenne:.4f} ({dt:.2f}s)")

    # 1. Sauvegarde des poids du modèle final
    chemin_final = output_dir / "final_model.pth"
    torch.save(modele.state_dict(), chemin_final)

    # Copie également sous model_weights.pth pour compatibilité
    torch.save(modele.state_dict(), output_dir / "model_weights.pth")
    print(f"\nPoids du modèle sauvegardés dans : {chemin_final.resolve()}")

    # 2. Sauvegarde de la courbe de perte
    chemin_plot = output_dir / "loss_train.png"
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), historique_perte, marker='o', color='royalblue', label="Perte train")
    plt.title("Courbe d'apprentissage - train.py", fontsize=12)
    plt.xlabel("Époque")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(chemin_plot)
    plt.close()
    print(f"Courbe d'apprentissage sauvegardée dans : {chemin_plot.resolve()}")

    # 3. Copie de classes.json dans le dossier de sortie
    if CLASSES_JSON_PATH.exists():
        import shutil
        shutil.copy2(CLASSES_JSON_PATH, output_dir / "classes.json")
        print(f"Fichier classes.json archivé dans : {output_dir.resolve()}")

    print("\n" + "=" * 65)
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train.py <racine des donnees> <dossier de sortie>")
        sys.exit(1)
    else:
        train(sys.argv[1], sys.argv[2])
