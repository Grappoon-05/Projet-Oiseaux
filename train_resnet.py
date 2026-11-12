import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

from CNN_base import OiseauxDataset, recup_classes, lister_images, charger_config
from select_device import device_config

# Configuration et appareil
config = charger_config("config.json")
data_cfg = config["data"]
train_cfg = config["training"]
out_cfg = config["output"]

appareil = device_config()

# Modèle ResNet18 avec Transfer Learning
def creer_modele_resnet(nb_classes=21):
    # 1. On charge le ResNet18 pré-entraîné sur la base ImageNet
    poids = models.ResNet18_Weights.DEFAULT
    modele = models.resnet18(weights=poids)

    # 2. On remplace la dernière couche linéaire pour qu'elle sorte nos 21 espèces d'oiseaux
    nb_entrees_fc = modele.fc.in_features
    modele.fc = nn.Linear(nb_entrees_fc, nb_classes)

    return modele


# Fonction d'entraînement pour ResNet18
def entrainer_resnet(modele, train_loader, criterion, optimizer, nb_epochs=15, device=appareil):
    modele.to(device)
    print(f"\n--- Début de l'entraînement ResNet18 ({nb_epochs} époques) ---")

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
    # 1. Classes cibles
    classes_cibles = recup_classes(data_cfg["classes_json_path"])
    print(f"Nombre d'espèces cibles : {len(classes_cibles)}")

    # 2. Chargement des images du train
    chemins_train, labels_train = lister_images(data_cfg["train_dir"], classes_cibles)
    print(f"Nombre d'images pour le train : {len(chemins_train)}")

    # 3. Transformations adaptées à ImageNet (224x224 pixels)
    taille_img = tuple(data_cfg.get("img_size", [224, 224]))
    transform_train = transforms.Compose([
        transforms.Resize(taille_img),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_train = OiseauxDataset(chemins_train, labels_train, transform=transform_train)
    batch_size = train_cfg.get("batch_size", 32)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    # 4. Création du modèle ResNet18
    modele = creer_modele_resnet(nb_classes=len(classes_cibles))
    print("\nModèle ResNet18 prêt avec la couche finale adaptée :")
    print(modele.fc)

    # Fonction de perte et optimiseur
    criterion = nn.CrossEntropyLoss()
    # Learning rate un peu plus faible pour préserver les poids ImageNet pré-entraînés
    lr = train_cfg.get("learning_rate", 0.0005)
    optimizer = optim.Adam(modele.parameters(), lr=lr)

    # 5. Entraînement
    nb_epochs = train_cfg.get("epochs", 15)
    modele_entraine, historique_pertes = entrainer_resnet(
        modele, train_loader, criterion, optimizer, nb_epochs=nb_epochs, device=appareil
    )

    # 6. Sauvegarde des poids
    fichier_poids = out_cfg.get("weights_path", "resnet18_birds.pth")
    torch.save(modele_entraine.state_dict(), fichier_poids)
    print(f"\nModèle ResNet18 enregistré dans : {fichier_poids}")

    # 7. Courbe de perte
    fichier_plot = out_cfg.get("loss_plot_path", "loss_resnet18.png")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, nb_epochs + 1), historique_pertes, label="Perte train ResNet18", color="blue")
    plt.xlabel("Époque")
    plt.ylabel("Perte")
    plt.title("Courbe d'apprentissage - ResNet18")
    plt.legend()
    plt.grid(True)
    plt.savefig(fichier_plot)
    print(f"Graphique de perte enregistré dans : {fichier_plot}")
