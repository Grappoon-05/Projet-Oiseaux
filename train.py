# Sources communautaires et documentation :
# 1. PyTorch Dataset & DataLoader Workflow :
#    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# 2. PyTorch Forums (Recommandation ImageFolder & DataLoader pour train/val séparés) :
#    https://discuss.pytorch.org/t/how-to-use-imagefolder-with-a-generator-or-train-test-split/35227
# 3. PyTorch Training Loop standard (Adam + CrossEntropyLoss) :
#    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import sys
import json
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from select_device import device_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def charger_image(path):
    return Image.open(path).convert("RGB")


def charger_classes(fichier_json="classes.json"):
    with open(fichier_json, "r", encoding="utf-8") as f:
        donnees = json.load(f)["classes"]
    classes = []
    for c in donnees:
        if c not in ("autre_oiseau", "rien", "incertain"):
            classes.append(c.strip().replace(" ", "_"))
    return sorted(list(set(classes)))


# Implémentation classique d'un Dataset d'images par dossiers (pattern PyTorch Forums)
class BirdDataset(Dataset):
    def __init__(self, root, classes, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.paths = []
        self.labels = []
        classe_vers_index = {nom: i for i, nom in enumerate(classes)}

        for nom in classes:
            dossier_classe = self.root / nom
            if not dossier_classe.is_dir():
                continue
            for fichier in sorted(dossier_classe.iterdir()):
                if fichier.suffix.lower() in IMAGE_EXTENSIONS:
                    self.paths.append(fichier)
                    self.labels.append(classe_vers_index[nom])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = charger_image(self.paths[index])
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label


def train(data_root, output_dir, epochs=10, batch_size=32, lr=0.0005):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    appareil = device_config()
    classes = charger_classes("classes.json")
    print(f"Entraînement sur {len(classes)} classes")

    # Transformations recommandées par torchvision.models pour ResNet18
    # Source : https://pytorch.org/vision/stable/models.html
    poids = models.ResNet18_Weights.DEFAULT
    transformations = poids.transforms()

    # Pattern Transfer Learning classique (Forum PyTorch / Tutoriel officiel)
    modele = models.resnet18(weights=poids)
    nb_features = modele.fc.in_features
    modele.fc = nn.Linear(nb_features, len(classes))
    modele = modele.to(appareil)

    dataset = BirdDataset(data_root, classes, transform=transformations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Images chargées : {len(dataset)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modele.parameters(), lr=lr)

    modele.train()
    for epoch in range(1, epochs + 1):
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
        print(f"Époque [{epoch:02d}/{epochs:02d}] - Perte: {perte_moyenne:.4f}")

    # Sauvegarde des poids
    chemin_modele = output_dir / "model_weights.pth"
    torch.save(modele.state_dict(), chemin_modele)
    print(f"Modèle sauvegardé dans : {chemin_modele}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train.py <racine des donnees> <dossier de sortie>")
        print("Lancement par défaut : ./dataset_split/train vers ./output ...")
        train("./dataset_split/train", "./output")
    else:
        train(sys.argv[1], sys.argv[2])
