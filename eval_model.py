import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from CNN_base import SimpleCNN, OiseauxDataset, get_target_classes, load_dataset_from_folder, load_config
from select_device import device_config

# ---------------------------------------------------------------------
# 1. Configuration et Device
# ---------------------------------------------------------------------
CONFIG = load_config("config.json")
data_cfg = CONFIG["data"]
train_cfg = CONFIG["training"]
out_cfg = CONFIG["output"]

device = device_config()

# ---------------------------------------------------------------------
# 2. Choix du dossier d'évaluation (val ou test)
# ---------------------------------------------------------------------
# Par défaut, on évalue sur le dossier 'val'
eval_folder = data_cfg.get("val_dir", "./dataset_split/val")
classes_cibles = get_target_classes(data_cfg["classes_json_path"])
eval_paths, eval_labels = load_dataset_from_folder(eval_folder, classes_cibles)

img_size = tuple(data_cfg.get("img_size", [128, 128]))
transform_eval = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_dataset = OiseauxDataset(eval_paths, eval_labels, transform=transform_eval)
batch_size = train_cfg.get("batch_size", 32)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

print(f"\nÉvaluation sur le dossier : '{eval_folder}'")
print(f"Nombre d'images à évaluer : {len(eval_dataset)} réparties sur {len(classes_cibles)} classes.")

# ---------------------------------------------------------------------
# 3. Chargement du modèle entraîné
# ---------------------------------------------------------------------
model = SimpleCNN(num_classes=len(classes_cibles), model_cfg=CONFIG["model"])
weights_path = out_cfg.get("weights_path", "baseline_cnn.pth")

model.load_state_dict(torch.load(weights_path, map_location=device))
model = model.to(device)
model.eval()
print(f"Modèle chargé avec succès depuis '{weights_path}'")

# ---------------------------------------------------------------------
# 4. Évaluation globale
# ---------------------------------------------------------------------
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
total_eval_loss = 0.0

class_correct = [0] * len(classes_cibles)
class_total = [0] * len(classes_cibles)

with torch.no_grad():
    for images, targets in eval_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_eval_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        for c, p in zip(targets, predicted):
            if c == p:
                class_correct[c.item()] += 1
            class_total[c.item()] += 1

avg_eval_loss = total_eval_loss / total
accuracy = 100.0 * correct / total

print("\n" + "=" * 50)
print(f"RÉSULTATS DE L'ÉVALUATION")
print("=" * 50)
print(f"Perte moyenne (Loss) : {avg_eval_loss:.4f}")
print(f"Précision globale (Accuracy) : {accuracy:.2f}% ({correct}/{total} correctes)")
print("=" * 50)

print("\nPrécision par espèce :")
for idx, name in enumerate(classes_cibles):
    if class_total[idx] > 0:
        acc_cls = 100.0 * class_correct[idx] / class_total[idx]
        print(f"  - {name:25} : {acc_cls:6.2f}% ({class_correct[idx]}/{class_total[idx]})")

# ---------------------------------------------------------------------
# 5. Visualisation d'échantillons
# ---------------------------------------------------------------------
def imshow(img, title=""):
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0.0, 1.0)
    plt.imshow(img)
    plt.title(title, fontsize=10)
    plt.axis("off")

plt.figure(figsize=(12, 6))
indices = random.sample(range(len(eval_dataset)), min(6, len(eval_dataset)))

for i, idx in enumerate(indices):
    img, true_label = eval_dataset[idx]
    input_tensor = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_label = output.argmax(dim=1).item()

    plt.subplot(2, 3, i + 1)
    is_correct = (pred_label == true_label)
    color_title = "green" if is_correct else "red"
    title = f"Vrai: {classes_cibles[true_label]}\nPrédit: {classes_cibles[pred_label]}"
    imshow(img, title)
    plt.title(title, color=color_title, fontsize=9)

plt.tight_layout()
plt.savefig("eval_predictions.png")
print("\nÉchantillons de prédictions enregistrés dans 'eval_predictions.png'")
