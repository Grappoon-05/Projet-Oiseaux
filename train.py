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

# Entraînement au format officiel demandé en Annexe A du sujet :
# python train.py <racine des donnees> <dossier de sortie>

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def load_image(path):
    return Image.open(path).convert("RGB")


def get_classes(classes_path="classes.json"):
    with open(classes_path, "r", encoding="utf-8") as f:
        raw = json.load(f)["classes"]
    classes = []
    for c in raw:
        if c not in ("autre_oiseau", "rien", "incertain"):
            classes.append(c.strip().replace(" ", "_"))
    return sorted(list(set(classes)))


class BirdDataset(Dataset):
    def __init__(self, root, classes, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.paths = []
        self.labels = []
        cls_to_idx = {name: i for i, name in enumerate(classes)}

        for name in classes:
            folder = self.root / name
            if not folder.is_dir():
                continue
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    self.paths.append(f)
                    self.labels.append(cls_to_idx[name])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_image(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def train(data_root, output_dir, epochs=10, batch_size=32, lr=0.0005):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = device_config()
    classes = get_classes("classes.json")
    print(f"Entraînement sur {len(classes)} classes")

    # Modèle ResNet18 exactement comme dans le sujet (page 5) :
    # models.ResNet18_Weights.DEFAULT.transforms()
    poids = models.ResNet18_Weights.DEFAULT
    transform = poids.transforms()

    modele = models.resnet18(weights=poids)
    modele.fc = torch.nn.Linear(modele.fc.in_features, len(classes))
    modele = modele.to(device)

    dataset = BirdDataset(data_root, classes, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Images chargées : {len(dataset)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modele.parameters(), lr=lr)

    modele.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = modele(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Époque [{epoch:02d}/{epochs:02d}] - Perte: {epoch_loss:.4f}")

    # Sauvegarde des poids finaux
    save_path = output_dir / "model_weights.pth"
    torch.save(modele.state_dict(), save_path)
    print(f"Modèle sauvegardé dans : {save_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Valeurs par défaut pratiques si lancé sans arguments
        print("Usage: python train.py <racine des donnees> <dossier de sortie>")
        print("Exécution avec dossiers par défaut (dataset_split/train vers ./output)...")
        data_dir = "./dataset_split/train"
        out_dir = "./output"
    else:
        data_dir = sys.argv[1]
        out_dir = sys.argv[2]

    train(data_dir, out_dir)
