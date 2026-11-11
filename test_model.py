import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from CNN_base import SimpleCNN, OiseauxDataset, recup_classes, lister_images, charger_config
from select_device import device_config

# Chargement de la configuration
config = charger_config("config.json")
data_cfg = config["data"]
train_cfg = config["training"]
out_cfg = config["output"]

appareil = device_config()

# Chargement du jeu de test final
dossier_test = data_cfg.get("test_dir", "./dataset_split/test")
classes_cibles = recup_classes(data_cfg["classes_json_path"])
chemins_test, labels_test = lister_images(dossier_test, classes_cibles)

taille_img = tuple(data_cfg.get("img_size", [128, 128]))
transform_test = transforms.Compose([
    transforms.Resize(taille_img),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_test = OiseauxDataset(chemins_test, labels_test, transform=transform_test)
batch_size = train_cfg.get("batch_size", 32)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

print(f"\nÉvaluation finale sur le jeu de test : '{dossier_test}'")
print(f"Nombre d'images : {len(dataset_test)} | Nombre d'espèces : {len(classes_cibles)}")

# Chargement du modèle CNN baseline
modele = SimpleCNN(nb_classes=len(classes_cibles), cfg_modele=config["model"])
fichier_poids = out_cfg.get("weights_path", "baseline_cnn.pth")

modele.load_state_dict(torch.load(fichier_poids, map_location=appareil))
modele = modele.to(appareil)
modele.eval()
print(f"Modèle chargé depuis '{fichier_poids}'")

# Boucle de test
nb_correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
perte_totale = 0.0

correct_par_classe = [0] * len(classes_cibles)
total_par_classe = [0] * len(classes_cibles)

with torch.no_grad():
    for images, cibles in test_loader:
        images, cibles = images.to(appareil), cibles.to(appareil)
        sorties = modele(images)
        perte = criterion(sorties, cibles)
        perte_totale += perte.item() * images.size(0)

        _, predictions = sorties.max(1)
        total += cibles.size(0)
        nb_correct += predictions.eq(cibles).sum().item()

        for c, p in zip(cibles, predictions):
            if c == p:
                correct_par_classe[c.item()] += 1
            total_par_classe[c.item()] += 1

perte_moyenne = perte_totale / total
precision_globale = 100.0 * nb_correct / total

print("\n" + "=" * 45)
print("RÉSULTATS DU TEST FINAL - BASELINE CNN")
print("=" * 45)
print(f"Perte moyenne (Loss) : {perte_moyenne:.4f}")
print(f"Précision globale (Accuracy) : {precision_globale:.2f}% ({nb_correct}/{total} correctes)")
print("=" * 45)

print("\nPrécision détaillée par espèce :")
for i, nom_espece in enumerate(classes_cibles):
    if total_par_classe[i] > 0:
        pct = 100.0 * correct_par_classe[i] / total_par_classe[i]
        print(f"  - {nom_espece:25} : {pct:5.1f}% ({correct_par_classe[i]}/{total_par_classe[i]})")

# Visualisation des résultats
def afficher_image(img, titre=""):
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0.0, 1.0)
    plt.imshow(img)
    plt.title(titre, fontsize=9)
    plt.axis("off")

plt.figure(figsize=(12, 6))
indices_hasard = random.sample(range(len(dataset_test)), min(6, len(dataset_test)))

for i, idx in enumerate(indices_hasard):
    img, vrai_label = dataset_test[idx]
    entree = img.unsqueeze(0).to(appareil)

    with torch.no_grad():
        pred_label = modele(entree).argmax(dim=1).item()

    plt.subplot(2, 3, i + 1)
    est_correct = (pred_label == vrai_label)
    couleur = "green" if est_correct else "red"
    titre = f"Vrai : {classes_cibles[vrai_label]}\nPrédit : {classes_cibles[pred_label]}"
    afficher_image(img, titre)
    plt.title(titre, color=couleur, fontsize=8)

plt.tight_layout()
plt.savefig("test_predictions.png")
print("\nExemples de prédictions enregistrés dans : test_predictions.png")
