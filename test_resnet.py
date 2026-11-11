import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix

from train_resnet import OiseauxDataset, recup_classes, lister_images, charger_config
from select_device import device_config

# ---------------------------------------------------------------------
# 1. Configuration et Device
# ---------------------------------------------------------------------
config = charger_config("config.json")
data_cfg = config["data"]
train_cfg = config["training"]
out_cfg = config["output"]

appareil = device_config()

# ---------------------------------------------------------------------
# 2. Chargement du jeu de test (224x224 pour ResNet18)
# ---------------------------------------------------------------------
dossier_test = data_cfg.get("test_dir", "./dataset_split/test")
classes_cibles = recup_classes(data_cfg["classes_json_path"])
chemins_test, labels_test = lister_images(dossier_test, classes_cibles)

taille_img = tuple(data_cfg.get("img_size", [224, 224]))
transform_test = transforms.Compose([
    transforms.Resize(taille_img),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_test = OiseauxDataset(chemins_test, labels_test, transform=transform_test)
batch_size = train_cfg.get("batch_size", 32)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

print(f"\nÉvaluation complète de ResNet18 sur : '{dossier_test}'")
print(f"Total images : {len(dataset_test)} | Nombre d'espèces : {len(classes_cibles)}")

# ---------------------------------------------------------------------
# 3. Chargement de ResNet18
# ---------------------------------------------------------------------
modele = models.resnet18(weights=None)
nb_entrees_fc = modele.fc.in_features
modele.fc = nn.Linear(nb_entrees_fc, len(classes_cibles))

fichier_poids = out_cfg.get("weights_path", "resnet18_birds.pth")
if not Path(fichier_poids).exists():
    print(f"Attention : fichier '{fichier_poids}' introuvable. Entraînez d'abord avec python3 train_resnet.py")
else:
    modele.load_state_dict(torch.load(fichier_poids, map_location=appareil))
    print(f"Poids chargés depuis '{fichier_poids}'")

modele = modele.to(appareil)
modele.eval()

# ---------------------------------------------------------------------
# 4. Calcul des prédictions et métriques complètes
# ---------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
perte_totale = 0.0

toutes_les_cibles = []
toutes_les_predictions = []

with torch.no_grad():
    for images, cibles in test_loader:
        images, cibles = images.to(appareil), cibles.to(appareil)
        sorties = modele(images)
        perte = criterion(sorties, cibles)
        perte_totale += perte.item() * images.size(0)

        _, predictions = sorties.max(1)

        toutes_les_cibles.extend(cibles.cpu().numpy())
        toutes_les_predictions.extend(predictions.cpu().numpy())

toutes_les_cibles = np.array(toutes_les_cibles)
toutes_les_predictions = np.array(toutes_les_predictions)

perte_moyenne = perte_totale / len(dataset_test)
accuracy_globale = 100.0 * np.sum(toutes_les_cibles == toutes_les_predictions) / len(dataset_test)

print("\n" + "=" * 60)
print("RAPPORT OFFICIEL DES PERFORMANCES - RESNET18 (ÉTAPE 2 / ÉTAPE 3)")
print("=" * 60)
print(f"Perte moyenne (Test Loss)   : {perte_moyenne:.4f}")
print(f"Précision globale (Accuracy) : {accuracy_globale:.2f}%")
print("=" * 60)

# Rapport détaillé Sklearn
print("\nRapport détaillé par espèce (Précision, Rappel, F1-Score) :\n")
print(classification_report(
    toutes_les_cibles,
    toutes_les_predictions,
    target_names=classes_cibles,
    digits=3,
    zero_division=0
))

# ---------------------------------------------------------------------
# 5. Matrice de Confusion
# ---------------------------------------------------------------------
cm = confusion_matrix(toutes_les_cibles, toutes_les_predictions)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
plt.title("Matrice de Confusion — ResNet18", fontsize=12)
plt.colorbar()
tick_marks = np.arange(len(classes_cibles))
plt.xticks(tick_marks, classes_cibles, rotation=90, fontsize=8)
plt.yticks(tick_marks, classes_cibles, fontsize=8)
plt.ylabel("Vraie espèce", fontsize=10)
plt.xlabel("Espèce prédite", fontsize=10)
plt.tight_layout()
plt.savefig("matrice_confusion_resnet.png")
print("Matrice de confusion sauvegardée dans : 'matrice_confusion_resnet.png'")

# ---------------------------------------------------------------------
# 6. Visualisation d'exemples de prédictions
# ---------------------------------------------------------------------
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
plt.savefig("test_predictions_resnet.png")
print("Exemples de prédictions enregistrés dans : 'test_predictions_resnet.png'")
