# Guide d'Utilisation du Projet — Reconnaissance d'Oiseaux

Ce document détaille le rôle de chaque fichier Python et de configuration du dépôt, ainsi que les commandes pour les exécuter pas à pas.

---

## 1. Vue d'ensemble des fichiers

| Fichier | Rôle | Étape du projet |
| :--- | :--- | :---: |
| `config.json` | Fichier central des paramètres (dossiers, taille d'images, époques, LR, sorties). | Global |
| `select_device.py` | Sélectionne automatiquement le meilleur matériel disponible (`mps`, `cuda`, `cpu`). | Global |
| `classes.json` | Définit la liste des classes d'oiseaux cibles. | Global |
| `split_dataset.py` | Découpe physiquement le jeu de données en 3 dossiers : `train/` (70%), `val/` (15%), `test/` (15%). | Préparation / Étape 0 |
| `CNN_base.py` | Entraîne le CNN simple fait maison ("from scratch") sur le dossier `train`. | Étape 1 |
| `eval_model.py` | Évalue le modèle CNN baseline sur le dossier de validation (`val`). | Étape 1 |
| `test_model.py` | Évalue le modèle CNN baseline sur le dossier de test final (`test`). | Étape 1 |
| `train_resnet.py` | Entraîne ResNet18 pré-entraîné (Transfer Learning) sur le dossier `train`. | Étape 2 |
| `test_resnet.py` | Évalue ResNet18 sur le dossier de test final (`test`). | Étape 2 |
| `yolo_crop.py` | Détecte les oiseaux via YOLO (`yolo11n.pt`) et recadre automatiquement les photos dans `dataset_cropped/`. | Étape 3 |
| `test_yolo.py` | Teste rapidement YOLO sur un échantillon d'images et génère les cadres dans `runs/yolo_test/`. | Étape 3 |

---

## 2. Déroulement chronologique et Commandes

### Étape 0 : Découpage des données
Avant d'entraîner un modèle, on sépare les photos en trois ensembles étanches :
```bash
python3 split_dataset.py
```
- **Entrée :** `Data_Hackaton/`
- **Sortie :** `dataset_split/` avec les sous-dossiers `train/` (719 images), `val/` (146 images) et `test/` (168 images).

---

### Étape 1 : Baseline CNN simple ("from scratch")

1. **Entraîner le CNN simple :**
   ```bash
   python3 CNN_base.py
   ```
   - Entraîne le modèle sur `dataset_split/train/`.
   - Sauvegarde les poids dans `baseline_cnn.pth`.
   - Génère la courbe d'apprentissage dans `loss_baseline.png`.

2. **Évaluer pendant le développement (Validation) :**
   ```bash
   python3 eval_model.py
   ```
   - Teste sur `dataset_split/val/` et génère `eval_predictions.png`.

3. **Mesurer le score officiel de la Baseline (Test) :**
   ```bash
   python3 test_model.py
   ```
   - Mesure l'accuracy finale sur `dataset_split/test/` et génère `test_predictions.png`.

---

### Étape 2 : Transfer Learning (ResNet18)

1. **Entraîner ResNet18 :**
   ```bash
   python3 train_resnet.py
   ```
   - Utilise les poids pré-entraînés ImageNet et adapte la couche finale aux 21 espèces.
   - Entraîne sur `dataset_split/train/` (résolution 224x224).
   - Sauvegarde les poids dans `resnet18_birds.pth` et la courbe dans `loss_resnet18.png`.

2. **Évaluer le score ResNet18 sur le test :**
   ```bash
   python3 test_resnet.py
   ```
   - Affiche l'accuracy globale et par espèce sur `dataset_split/test/`.
   - Génère `test_predictions_resnet.png`.

---

### Étape 3 : Détecter avant de classer (YOLO Crop)

1. **Tester la détection YOLO visuellement :**
   ```bash
   python3 test_yolo.py
   ```
   - Détecte la classe oiseau sur un échantillon de test et enregistre les images avec boîtes vertes dans `runs/yolo_test/`.

2. **Générer le jeu de données recadré (Crops) :**
   ```bash
   python3 yolo_crop.py
   ```
   - Parcourt `dataset_split/` (train, val, test).
   - Découpe chaque oiseau avec une marge de 10%.
   - Si aucun oiseau n'est détecté, conserve l'image en repli (*fallback*).
   - Crée le dossier `dataset_cropped/`.

3. **Entraîner ResNet18 sur les oiseaux recadrés :**
   Dans `config.json`, modifier les chemins vers le dossier recadré :
   ```json
   "train_dir": "./dataset_cropped/train",
   "val_dir": "./dataset_cropped/val",
   "test_dir": "./dataset_cropped/test",
   "weights_path": "resnet18_yolo_crop.pth",
   "loss_plot_path": "loss_resnet18_yolo.png"
   ```
   Puis lancer :
   ```bash
   python3 train_resnet.py
   python3 test_resnet.py
   ```

---

## 3. Configuration centralisée (`config.json`)

Toutes les options principales se modifient dans `config.json` sans avoir à toucher au code :
- `dataset_root` / `train_dir` / `val_dir` / `test_dir` : Dossiers des images.
- `img_size` : Résolution d'entrée (`[128, 128]` pour CNN simple, `[224, 224]` pour ResNet).
- `epochs` : Nombre d'époques d'entraînement.
- `batch_size` : Taille des lots (ex: 32).
- `learning_rate` : Taux d'apprentissage (`0.001` pour CNN simple, `0.0005` pour ResNet).
- `weights_path` : Nom du fichier de sauvegarde des poids (`.pth`).
