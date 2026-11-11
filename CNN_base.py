import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from select_device import device_config

# On charge le fichier de configuration json
def charger_config(nom_fichier="config.json"):
    with open(nom_fichier, "r", encoding="utf-8") as f:
        return json.load(f)

config = charger_config("config.json")
appareil = device_config()

# Modèle CNN simple fait maison pour la baseline
class SimpleCNN(nn.Module):
    def __init__(self, nb_classes=21, cfg_modele=None):
        super(SimpleCNN, self).__init__()
        if cfg_modele is None:
            cfg_modele = config["model"]

        nb_canaux = cfg_modele.get("num_channels", 3)
        filtres1 = cfg_modele.get("conv1_filters", 32)
        filtres2 = cfg_modele.get("conv2_filters", 64)
        filtres3 = cfg_modele.get("conv3_filters", 128)
        taille_noyau = cfg_modele.get("kernel_size", 3)
        padding = cfg_modele.get("padding", 1)

        # Partie convolution (extraction des formes, contours, textures)
        self.convolutions = nn.Sequential(
            # Bloc 1
            nn.Conv2d(nb_canaux, filtres1, kernel_size=taille_noyau, padding=padding),
            nn.BatchNorm2d(filtres1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloc 2
            nn.Conv2d(filtres1, filtres2, kernel_size=taille_noyau, padding=padding),
            nn.BatchNorm2d(filtres2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Bloc 3
            nn.Conv2d(filtres2, filtres3, kernel_size=taille_noyau, padding=padding),
            nn.BatchNorm2d(filtres3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Partie classification pour sortir le score de chaque classe
        self.classifieur = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(filtres3, nb_classes)
        )

    def forward(self, x):
        features = self.convolutions(x)
        sorties = self.classifieur(features)
        return sorties


# Extensions d'images prises en compte
EXTENSIONS_IMAGES = {".jpg", ".jpeg", ".png"}

def charger_image(chemin):
    # Ouvre l'image et la convertit en RGB (pour éviter les erreurs d'images noir et blanc)
    return Image.open(chemin).convert("RGB")


def recup_classes(chemin_json):
    # Lit le fichier classes.json et garde seulement les vraies espèces d'oiseaux
    with open(chemin_json, "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    classes_propres = []
    for c in classes:
        if c not in ("autre_oiseau", "rien", "incertain"):
            classes_propres.append(c.strip().replace(" ", "_"))
    return sorted(list(set(classes_propres)))


def lister_images(dossier, classes_cibles):
    # Parcourt les sous-dossiers et renvoie les chemins des images et leurs labels (0, 1, 2...)
    dossier_path = Path(dossier)
    chemins, labels = [], []
    classe_vers_id = {nom: i for i, nom in enumerate(classes_cibles)}

    for nom_classe in classes_cibles:
        dossier_classe = dossier_path / nom_classe
        if not dossier_classe.is_dir():
            continue

        for fichier in sorted(dossier_classe.iterdir()):
            if fichier.suffix.lower() in EXTENSIONS_IMAGES:
                chemins.append(fichier)
                labels.append(classe_vers_id[nom_classe])

    return chemins, labels


# Dataset PyTorch pour charger les images à la volée
class OiseauxDataset(Dataset):
    def __init__(self, chemins, labels, transform=None):
        self.chemins = chemins
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.chemins)

    def __getitem__(self, index):
        image = charger_image(self.chemins[index])
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label


# Fonction d'entraînement du modèle
def entrainer(modele, train_loader, criterion, optimizer, nb_epochs=15, device=appareil):
    modele.to(device)
    print(f"\n--- Début de l'entraînement ({nb_epochs} époques) ---")

    pertes_train = []

    for epoch in range(1, nb_epochs + 1):
        temps_debut = time.time()
        modele.train()
        perte_totale = 0.0

        for images, cibles in train_loader:
            images = images.to(device)
            cibles = cibles.to(device)

            optimizer.zero_grad()
            predictions = modele(images)
            perte = criterion(predictions, cibles)
            perte.backward()
            optimizer.step()

            perte_totale += perte.item() * images.size(0)

        perte_moyenne = perte_totale / len(train_loader.dataset)
        pertes_train.append(perte_moyenne)
        duree = time.time() - temps_debut

        print(f"Époque [{epoch:02d}/{nb_epochs:02d}] - Perte train: {perte_moyenne:.4f} - Temps: {duree:.2f}s")

    return modele, pertes_train


if __name__ == '__main__':
    data_cfg = config["data"]
    train_cfg = config["training"]
    out_cfg = config["output"]

    # 1. On récupère les classes
    classes_cibles = recup_classes(data_cfg["classes_json_path"])
    print(f"Nombre d'espèces cibles : {len(classes_cibles)}")

    # 2. On charge les images du dossier d'entraînement
    chemins_train, labels_train = lister_images(data_cfg["train_dir"], classes_cibles)
    print(f"Nombre d'images pour le train : {len(chemins_train)}")

    # 3. Transformations appliquées aux images (redimensionnement et data augmentation basique)
    taille_img = tuple(data_cfg.get("img_size", [128, 128]))
    transform_train = transforms.Compose([
        transforms.Resize(taille_img),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_train = OiseauxDataset(chemins_train, labels_train, transform=transform_train)

    batch_size = train_cfg.get("batch_size", 32)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    # 4. Création du modèle
    modele = SimpleCNN(nb_classes=len(classes_cibles), cfg_modele=config["model"])
    print("\nArchitecture du modèle :")
    print(modele)

    # Fonction de perte pour la classification et optimiseur Adam
    criterion = nn.CrossEntropyLoss()
    lr = train_cfg.get("learning_rate", 0.001)
    optimizer = optim.Adam(modele.parameters(), lr=lr)

    # 5. Lancement de l'entraînement
    nb_epochs = train_cfg.get("epochs", 15)
    modele_entraine, historique_pertes = entrainer(
        modele, train_loader, criterion, optimizer, nb_epochs=nb_epochs, device=appareil
    )

    # 6. Sauvegarde des poids du modèle
    fichier_poids = out_cfg.get("weights_path", "baseline_cnn.pth")
    torch.save(modele_entraine.state_dict(), fichier_poids)
    print(f"\nPoids du modèle enregistrés dans : {fichier_poids}")

    # 7. Affichage et sauvegarde de la courbe de perte
    fichier_plot = out_cfg.get("loss_plot_path", "loss_baseline.png")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, nb_epochs + 1), historique_pertes, label="Perte train", color="blue")
    plt.xlabel("Époque")
    plt.ylabel("Perte (CrossEntropy)")
    plt.title("Courbe d'apprentissage - Baseline CNN")
    plt.legend()
    plt.grid(True)
    plt.savefig(fichier_plot)
    print(f"Graphique de perte enregistré dans : {fichier_plot}")
