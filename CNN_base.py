import time
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from select_device import device_config

# =====================================================================
# 1. Configuration et Device
# =====================================================================
def load_config(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

CONFIG = load_config("config.json")
device = device_config()

# =====================================================================
# 2. Architecture SimpleCNN
# =====================================================================
class SimpleCNN(nn.Module):
    """
    CNN simple 'from scratch' pour la classification d'espèces d'oiseaux.
    Architecture 3 blocs Conv2d + BatchNorm2d + ReLU + MaxPool2d + Dropout.
    """
    def __init__(self, num_classes=21, model_cfg=None):
        super(SimpleCNN, self).__init__()
        if model_cfg is None:
            model_cfg = CONFIG["model"]

        in_channels = model_cfg.get("num_channels", 3)
        c1 = model_cfg.get("conv1_filters", 32)
        c2 = model_cfg.get("conv2_filters", 64)
        c3 = model_cfg.get("conv3_filters", 128)
        k = model_cfg.get("kernel_size", 3)
        p = model_cfg.get("padding", 1)

        self.encoder = nn.Sequential(
            # Bloc 1 (128x128 -> 64x64)
            nn.Conv2d(in_channels, c1, kernel_size=k, padding=p),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloc 2 (64x64 -> 32x32)
            nn.Conv2d(c1, c2, kernel_size=k, padding=p),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloc 3 (32x32 -> 16x16)
            nn.Conv2d(c2, c3, kernel_size=k, padding=p),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(c3, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        out = self.classifier(features)
        return out


# =====================================================================
# 3. Chargement des données par dossier
# =====================================================================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def load_image(path):
    return Image.open(path).convert("RGB")


def get_target_classes(classes_json_path):
    """Lit classes.json et filtre sur les espèces réelles."""
    with open(classes_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["classes"]

    target = []
    for c in data:
        if c not in ("autre_oiseau", "rien", "incertain"):
            target.append(c.strip().replace(" ", "_"))
    return sorted(list(set(target)))


def load_dataset_from_folder(folder_path, target_classes):
    """Charge les chemins et labels directement depuis un dossier."""
    root = Path(folder_path)
    paths, labels = [], []
    class_to_idx = {name: idx for idx, name in enumerate(target_classes)}

    for class_name in target_classes:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue

        for file in sorted(class_dir.iterdir()):
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(file)
                labels.append(class_to_idx[class_name])

    return paths, labels


class OiseauxDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = load_image(self.paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# =====================================================================
# 4. Fonction d'entraînement (uniquement sur Train)
# =====================================================================
def train_model(model, train_loader, criterion, optimizer, num_epochs=15, device=device):
    model.to(device)
    print(f"\n--- Début de l'entraînement ({num_epochs} époques) ---")

    train_losses = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        running_train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - "
              f"Train Loss: {epoch_train_loss:.4f} - Durée: {elapsed:.2f}s")

    return model, train_losses


# =====================================================================
# 5. Exécution principale
# =====================================================================
if __name__ == '__main__':
    data_cfg = CONFIG["data"]
    train_cfg = CONFIG["training"]
    out_cfg = CONFIG["output"]

    # 1. Classes cibles
    classes_cibles = get_target_classes(data_cfg["classes_json_path"])
    print(f"Nombre de classes cibles : {len(classes_cibles)}")

    # 2. Chargement du dossier physique Train uniquement
    train_paths, train_labels = load_dataset_from_folder(data_cfg["train_dir"], classes_cibles)
    print(f"Images chargées - Train : {len(train_paths)}")

    # 3. Transformations
    img_size = tuple(data_cfg.get("img_size", [128, 128]))
    
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = OiseauxDataset(train_paths, train_labels, transform=transform_train)

    batch_size = train_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Modèle
    model = SimpleCNN(num_classes=len(classes_cibles), model_cfg=CONFIG["model"])
    print("\nModèle instancié :")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.get("learning_rate", 0.001))

    # 5. Entraînement
    epochs = train_cfg.get("epochs", 15)
    trained_model, train_losses = train_model(
        model, train_loader, criterion, optimizer, num_epochs=epochs, device=device
    )

    # 6. Sauvegarde des poids
    weights_path = out_cfg.get("weights_path", "baseline_cnn.pth")
    torch.save(trained_model.state_dict(), weights_path)
    print(f"\nModèle entraîné sauvegardé dans '{weights_path}'.")

    # 7. Tracé de la courbe de perte d'entraînement
    plot_path = out_cfg.get("loss_plot_path", "loss_baseline.png")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="blue")
    plt.xlabel("Époques")
    plt.ylabel("Perte")
    plt.title("Évolution de la perte d'entraînement (Train Loss)")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Graphique sauvegardé dans '{plot_path}'.")
