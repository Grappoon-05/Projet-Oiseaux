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


# Dataset pour charger les images depuis une arborescence de dossiers de classes
class BirdDataset(Dataset):
    def __init__(self, root, classes, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.paths = []
        self.labels = []
        classe_vers_index = {nom: i for i, nom in enumerate(classes)}

        # Cas 1 : root pointe directement vers un dossier contenant des sous-dossiers de classes
        # ou un dossier qui contient un sous-dossier train/ contenant les classes
        dossier_source = self.root
        if (self.root / "train").is_dir():
            dossier_source = self.root / "train"

        # Recherche récursive de chaque classe dans l'arborescence fournie
        for nom in classes:
            # Recherche directe sous le dossier source
            dossier_classe = dossier_source / nom
            if dossier_classe.is_dir():
                for fichier in sorted(dossier_classe.iterdir()):
                    if fichier.suffix.lower() in IMAGE_EXTENSIONS:
                        self.paths.append(fichier)
                        self.labels.append(classe_vers_index[nom])
            else:
                # Recherche récursive au cas où les classes sont dans des sous-dossiers (ex: hackaton_split/train/)
                for sous_dossier in dossier_source.rglob(nom):
                    if sous_dossier.is_dir():
                        for fichier in sorted(sous_dossier.iterdir()):
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

    # Transformations officielles pour ResNet18
    poids = models.ResNet18_Weights.DEFAULT
    transformations = poids.transforms()

    # Modèle ResNet18 pré-entraîné adapté aux classes d'oiseaux
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

    # Sauvegarde des poids dans le dossier de sortie demandé
    chemin_modele = output_dir / "final_model.pth"
    torch.save(modele.state_dict(), chemin_modele)
    
    # Copie également sous model_weights.pth pour compatibilité
    torch.save(modele.state_dict(), output_dir / "model_weights.pth")
    print(f"Modèle sauvegardé avec succès dans : {chemin_modele}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train.py <racine des donnees> <dossier de sortie>")
        print("Lancement par défaut : ./dataset_split/train vers ./output ...")
        train("./dataset_split/train", "./output")
    else:
        train(sys.argv[1], sys.argv[2])
