import time
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from selection_device import device_config

# =====================================================================
# 1. Chargement du fichier de configuration JSON
# =====================================================================
def load_config(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

CONFIG = load_config("config.json")

# Choix automatique du device (MPS sur Mac Apple Silicon, CUDA, ou CPU)
device = device_config()

# =====================================================================
# 2. Architecture SimpleCNN paramétrée par la configuration
# =====================================================================
class SimpleCNN(nn.Module):
    """
    CNN simple 'from scratch' pour la classification d'espèces d'oiseaux (TP3).
    Les hyperparamètres sont lus depuis le config.json.
    """
    def __init__(self, num_classes=21, model_cfg=None):
        super(SimpleCNN, self).__init__()
        if model_cfg is None:
            model_cfg = CONFIG["model"]

        in_channels = model_cfg.get("num_channels", 3)
        c1 = model_cfg.get("conv1_filters", 32)
        c2 = model_cfg.get("conv2_filters", 64)
        k = model_cfg.get("kernel_size", 3)
        p = model_cfg.get("padding", 1)

        # Structure convolutive directe du TP3
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=k, padding=p),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c1, c2, kernel_size=k, padding=p),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Tête de classification vers le nombre de classes d'oiseaux
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c2, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        out = self.classifier(features)
        return out


# =====================================================================
# 3. Chargement des données (Data_Hackaton + classes.json)
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


def list_images(root, target_classes):
    """Parcourt le dossier pour les classes cibles comme dans load_dataset.py."""
    root = Path(root)
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
# 4. Fonction d'entraînement (Structure du TP3)
# =====================================================================
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device=device):
    """
    Entraîne le modèle, affiche la loss et l'accuracy à chaque epoch.
    """
    model.to(device)
    print(f"\n--- Début de l'entraînement ({num_epochs} époques) ---")

    train_losses = []
    test_losses = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # 1. Phase d'entraînement
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

        # 2. Évaluation sur l'ensemble de test
        model.eval()
        running_test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item() * images.size(0)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)
        accuracy = 100.0 * correct / total
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - "
              f"Train Loss: {epoch_train_loss:.4f} - Test Loss: {epoch_test_loss:.4f} - "
              f"Accuracy Test: {accuracy:.2f}% - Durée: {elapsed:.2f}s")

    return model, train_losses, test_losses


# =====================================================================
# 5. Exécution principale utilisant la configuration
# =====================================================================
if __name__ == '__main__':
    data_cfg = CONFIG["data"]
    train_cfg = CONFIG["training"]
    out_cfg = CONFIG["output"]

    # 1. Classes et images
    classes_cibles = get_target_classes(data_cfg["classes_json_path"])
    print(f"Nombre de classes cibles : {len(classes_cibles)}")

    paths, labels = list_images(data_cfg["dataset_root"], classes_cibles)
    print(f"Nombre total d'images chargées : {len(paths)}")

    # 2. Transformations d'images (taille issue de la config)
    img_size = tuple(data_cfg.get("img_size", [128, 128]))
    transform_img = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])

    # 3. Split Train / Test
    dataset = OiseauxDataset(paths, labels, transform=transform_img)
    split_ratio = data_cfg.get("train_split_ratio", 0.8)
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size

    seed = data_cfg.get("seed", 42)
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    batch_size = train_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Images - Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # 4. Modèle
    model = SimpleCNN(num_classes=len(classes_cibles), model_cfg=CONFIG["model"])
    print("\nModèle instancié avec la configuration :")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.get("learning_rate", 0.001))

    # 5. Entraînement
    epochs = train_cfg.get("epochs", 10)
    trained_model, train_losses, test_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=epochs, device=device
    )

    # 6. Sauvegarde des poids
    weights_path = out_cfg.get("weights_path", "baseline_cnn.pth")
    torch.save(trained_model.state_dict(), weights_path)
    print(f"\nModèle sauvegardé dans '{weights_path}'.")

    # 7. Sauvegarde du graphique des pertes
    plot_path = out_cfg.get("loss_plot_path", "loss_baseline.png")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss", color="orange")
    plt.xlabel("Époques")
    plt.ylabel("Perte")
    plt.title("Évolution des pertes (Baseline TP3)")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Graphique sauvegardé dans '{plot_path}'.")
