import argparse
import csv
import json
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report

from select_device import device_config

# ---------------------------------------------------------------------
# Arguments CLI (compatible Mac local et Google Colab)
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Expériences globales du projet oiseaux.")
parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("/Users/grappoon/Desktop/IUT/L2/S3/R3.14 - IA/projex"),
    help="Dossier contenant hackaton_split, laurens_split et combined_split.",
)
args, _ = parser.parse_known_args()

PROJEX_DIR = args.data_dir
DATASETS = {
    "hackaton": {
        "train": PROJEX_DIR / "hackaton_split" / "train",
        "test": PROJEX_DIR / "hackaton_split" / "test"
    },
    "laurens": {
        "train": PROJEX_DIR / "laurens_split" / "train",
        "test": PROJEX_DIR / "laurens_split" / "test"
    },
    "combine": {
        "train": PROJEX_DIR / "combined_split" / "train",
        "test": PROJEX_DIR / "combined_split" / "test"
    }
}

OUTPUT_CSV = PROJEX_DIR / "TABLEAU_DES_GAINS.csv"
OUTPUT_CSV_LOCAL = Path("TABLEAU_DES_GAINS.csv")

MODELS_DIR = PROJEX_DIR / "modeles_entraines"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = PROJEX_DIR / "graphiques_resultats"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR_LOCAL = Path("graphiques_resultats")
PLOTS_DIR_LOCAL.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_all_classes():
    classes_path = Path("classes.json")
    if not classes_path.exists():
        classes_path = PROJEX_DIR / "classes.json"
    with open(classes_path, "r", encoding="utf-8") as f:
        raw = json.load(f)["classes"]
    classes = []
    for c in raw:
        if c not in ("autre_oiseau", "incertain"):
            classes.append(c.strip().replace(" ", "_"))
    return sorted(list(set(classes)))


CLASSES = get_all_classes()
CLS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


class GenericBirdDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None, crop_with_yolo=False, yolo_model=None, device="cpu"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.crop_with_yolo = crop_with_yolo
        self.yolo_model = yolo_model
        self.device = device
        self.samples = []

        for cls_name in classes:
            folder = self.root_dir / cls_name
            if not folder.is_dir():
                continue
            for img_path in sorted(folder.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, CLS_TO_IDX[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.crop_with_yolo and self.yolo_model is not None:
            res = self.yolo_model(str(path), classes=[14], verbose=False, device=self.device)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                idx_max = int(res.boxes.conf.argmax())
                x1, y1, x2, y2 = res.boxes.xyxy[idx_max].tolist()
                w, h = img.size
                pad_w = (x2 - x1) * 0.10
                pad_h = (y2 - y1) * 0.10
                img = img.crop((
                    max(0, int(x1 - pad_w)),
                    max(0, int(y1 - pad_h)),
                    min(w, int(x2 + pad_w)),
                    min(h, int(y2 + pad_h))
                ))

        if self.transform:
            img = self.transform(img)

        return img, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_resnet(num_classes=len(CLASSES)):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def evaluate_dataset(model, test_dataset, device):
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    loss_tot = 0.0
    crit = nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_tot += crit(out, y).item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    if total == 0:
        return 0.0, 0.0, np.array([]), np.array([])
    acc = 100.0 * correct / total
    avg_loss = loss_tot / total
    return acc, avg_loss, np.array(all_preds), np.array(all_targets)


def plot_and_save_confusion_matrix(targets, preds, classes, title, output_path):
    if len(targets) == 0:
        return
    cm = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=11)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=7)
    plt.yticks(tick_marks, classes, fontsize=7)
    plt.ylabel("Vraie classe", fontsize=9)
    plt.xlabel("Classe prédite", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_and_eval_run(etape_nom, source_train_nom, model_type, img_size, use_yolo=False, epochs=10, lr=0.001):
    device = device_config()
    device_str = str(device)
    print(f"\n==================================================")
    print(f"RUN : {etape_nom} | Source Train : {source_train_nom} | Modele : {model_type}")
    print(f"==================================================")

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    yolo_mod = YOLO("yolo11n.pt") if use_yolo else None

    # Datasets
    train_dir = DATASETS[source_train_nom]["train"]
    ds_train = GenericBirdDataset(train_dir, CLASSES, transform=transform_train, crop_with_yolo=use_yolo, yolo_model=yolo_mod, device=device_str)
    train_loader = DataLoader(ds_train, batch_size=32, shuffle=True)
    print(f"Images d'entraînement chargées : {len(ds_train)}")

    # Datasets de test pour suivi de perte à chaque époque
    ds_test_hack = GenericBirdDataset(DATASETS["hackaton"]["test"], CLASSES, transform=transform_test, crop_with_yolo=use_yolo, yolo_model=yolo_mod, device=device_str)
    ds_test_lau = GenericBirdDataset(DATASETS["laurens"]["test"], CLASSES, transform=transform_test, crop_with_yolo=use_yolo, yolo_model=yolo_mod, device=device_str)

    # Modèle
    if model_type == "CNN_base":
        modele = SimpleCNN(num_classes=len(CLASSES)).to(device)
    else:
        modele = get_resnet(num_classes=len(CLASSES)).to(device)

    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(modele.parameters(), lr=lr)

    # Historiques pour les courbes de perte Train vs Test
    train_losses = []
    test_losses_hack = []
    test_losses_lau = []

    # Boucle d'entraînement
    for ep in range(1, epochs + 1):
        modele.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = modele(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)

        epoch_train_loss = running_loss / max(1, len(ds_train))
        train_losses.append(epoch_train_loss)

        # Calcul des pertes de test à cette époque pour la courbe
        _, loss_hack_ep, _, _ = evaluate_dataset(modele, ds_test_hack, device)
        _, loss_lau_ep, _, _ = evaluate_dataset(modele, ds_test_lau, device)
        test_losses_hack.append(loss_hack_ep)
        test_losses_lau.append(loss_lau_ep)

        if ep % 2 == 0 or ep == epochs:
            print(f"  Époque [{ep:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.4f} | Test Loss Hack: {loss_hack_ep:.4f} | Test Loss Lau: {loss_lau_ep:.4f}")

    # Sauvegarde des poids
    nom_base = f"{etape_nom}_{source_train_nom}_{model_type}"
    poids_file = MODELS_DIR / f"{nom_base}.pth"
    torch.save(modele.state_dict(), poids_file)

    # 1. Tracé de la courbe de perte Train vs Tests (Hackaton et Laurens)
    fig_loss = PLOTS_DIR / f"courbe_loss_{nom_base}.png"
    fig_loss_local = PLOTS_DIR_LOCAL / f"courbe_loss_{nom_base}.png"
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="blue", linewidth=2)
    plt.plot(range(1, epochs + 1), test_losses_hack, label="Test Loss (Hackaton)", color="orange", linestyle="--")
    plt.plot(range(1, epochs + 1), test_losses_lau, label="Test Loss (Laurens)", color="green", linestyle=":")
    plt.xlabel("Époques")
    plt.ylabel("Perte (CrossEntropy)")
    plt.title(f"Courbe de Perte — {nom_base}")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_loss)
    plt.savefig(fig_loss_local)
    plt.close()

    # 2. Évaluation finale et Matrices de confusion
    acc_hack, loss_hack, preds_hack, targets_hack = evaluate_dataset(modele, ds_test_hack, device)
    acc_lau, loss_lau, preds_lau, targets_lau = evaluate_dataset(modele, ds_test_lau, device)

    # Matrice de confusion sur test Hackaton
    cm_hack_path = PLOTS_DIR / f"matrice_confusion_hack_{nom_base}.png"
    plot_and_save_confusion_matrix(targets_hack, preds_hack, CLASSES, f"Matrice de Confusion (Hackaton Test) — {nom_base}", cm_hack_path)
    plot_and_save_confusion_matrix(targets_hack, preds_hack, CLASSES, f"Matrice de Confusion (Hackaton Test) — {nom_base}", PLOTS_DIR_LOCAL / f"matrice_confusion_hack_{nom_base}.png")

    # Matrice de confusion sur test Laurens (Piège photo)
    cm_lau_path = PLOTS_DIR / f"matrice_confusion_lau_{nom_base}.png"
    plot_and_save_confusion_matrix(targets_lau, preds_lau, CLASSES, f"Matrice de Confusion (Laurens Test) — {nom_base}", cm_lau_path)
    plot_and_save_confusion_matrix(targets_lau, preds_lau, CLASSES, f"Matrice de Confusion (Laurens Test) — {nom_base}", PLOTS_DIR_LOCAL / f"matrice_confusion_lau_{nom_base}.png")

    print(f"\nRésultats finaux :")
    print(f"  Acc Test Hackaton : {acc_hack:.2f}% (Loss: {loss_hack:.4f})")
    print(f"  Acc Test Laurens  : {acc_lau:.2f}% (Loss: {loss_lau:.4f})")
    print(f"  Courbe de perte   : {fig_loss}")
    print(f"  Matrices de conf  : {cm_hack_path.name} & {cm_lau_path.name}")

    return {
        "etape": etape_nom,
        "modele": model_type,
        "source_train": source_train_nom,
        "acc_hackaton_test": round(acc_hack, 2),
        "loss_hackaton_test": round(loss_hack, 4),
        "acc_laurens_test": round(acc_lau, 2),
        "loss_laurens_test": round(loss_lau, 4),
        "courbe_loss": str(fig_loss),
        "matrice_confusion_laurens": str(cm_lau_path),
        "poids_sauvegardes": str(poids_file)
    }


def main():
    resultats = []

    # ÉTAPE 1 : CNN BASELINE
    r1 = train_and_eval_run("Etape1_Baseline", "hackaton", "CNN_base", img_size=128, epochs=10, lr=0.001)
    resultats.append(r1)
    r2 = train_and_eval_run("Etape1_Baseline", "laurens", "CNN_base", img_size=128, epochs=10, lr=0.001)
    resultats.append(r2)
    r3 = train_and_eval_run("Etape1_Baseline", "combine", "CNN_base", img_size=128, epochs=10, lr=0.001)
    resultats.append(r3)

    # ÉTAPE 2 : RESNET18 TRANSFER LEARNING
    r4 = train_and_eval_run("Etape2_ResNet18", "hackaton", "ResNet18", img_size=224, epochs=10, lr=0.0005)
    resultats.append(r4)
    r5 = train_and_eval_run("Etape2_ResNet18", "laurens", "ResNet18", img_size=224, epochs=10, lr=0.0005)
    resultats.append(r5)
    r6 = train_and_eval_run("Etape2_ResNet18", "combine", "ResNet18", img_size=224, epochs=10, lr=0.0005)
    resultats.append(r6)

    # ÉTAPE 3 : YOLO CROP + RESNET18
    r7 = train_and_eval_run("Etape3_YOLO_ResNet", "hackaton", "ResNet18", img_size=224, use_yolo=True, epochs=10, lr=0.0005)
    resultats.append(r7)
    r8 = train_and_eval_run("Etape3_YOLO_ResNet", "laurens", "ResNet18", img_size=224, use_yolo=True, epochs=10, lr=0.0005)
    resultats.append(r8)
    r9 = train_and_eval_run("Etape3_YOLO_ResNet", "combine", "ResNet18", img_size=224, use_yolo=True, epochs=10, lr=0.0005)
    resultats.append(r9)

    # Sauvegarde CSV
    champs = [
        "etape", "modele", "source_train", "acc_hackaton_test", "loss_hackaton_test",
        "acc_laurens_test", "loss_laurens_test", "courbe_loss", "matrice_confusion_laurens", "poids_sauvegardes"
    ]
    for dest in [OUTPUT_CSV, OUTPUT_CSV_LOCAL]:
        with open(dest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=champs)
            writer.writeheader()
            writer.writerows(resultats)

    print("\n" + "=" * 60)
    print("TOUTES LES EXPÉRIENCES SONT TERMINÉES !")
    print(f"Tableau complet sauvegardé dans : {OUTPUT_CSV}")
    print(f"Graphiques (courbes et matrices) dans : {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
