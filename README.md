# Projet Reconnaissance d'Oiseaux — RobIA LP2 (R3.14 Deep Learning)

Ce dépôt contient la solution complète pour identifier les espèces d'oiseaux visitant une mangeoire de jardin à partir des images brutes d'un piège photo.

---

## 1. Prérequis & Installation

```bash
# Cloner le dépôt
git clone https://github.com/Grappoon-05/Projet-Oiseaux.git
cd Projet-Oiseaux

# Installer les dépendances
python -m pip install -r requirements.txt
```

---

## 2. Commandes

### 2.1. Entraînement
Pour entraîner le modèle sur un jeu de données (les images organisées en sous-dossiers par classe) :

```bash
python train.py <racine_des_donnees> <dossier_de_sortie>
```
*Exemple :*
```bash
python train.py ./dataset_split/train ./output
```

### 2.2. Inférence
Pour lancer la prédiction sur un dossier d'images brutes et générer le fichier `predictions.csv` :

```bash
python predict.py <dossier_images> <dossier_de_sortie>
```
*Exemple :*
```bash
python predict.py ./test_images ./output
```

Le script utilise automatiquement les poids du modèle final (`final_model.pth`) et la détection YOLO pour générer `predictions.csv` avec l'en-tête exact :
```csv
image_id,sequence_id,label,confidence
```

---

## 3. Périmètre des Classes (`classes.json`)
Le périmètre d'identification est déclaré dans `classes.json` avec les noms latins normalisés de l'Annexe B :
- `Streptopelia_decaocto`, `Erithacus_rubecula`, `Fringilla_coelebs`, `Parus_major`, `Turdus_merula`, `Cyanistes_caeruleus`, `Pica_pica`, `Passer_domesticus`, `Sturnus_vulgaris`, `Chloris_chloris`, `Carduelis_carduelis`, `Upupa_epops`, `Garrulus_glandarius`, `Serinus_serinus`, `Sylvia_atricapilla`, `Columba_palumbus`.
- **Classes spéciales :** `"rien"` (déclenchement à vide) et `"autre_oiseau"` (espèce hors périmètre).

---

## 4. Organisation du Projet

- `train.py` : Script officiel d'entraînement.
- `predict.py` : Script officiel d'inférence autonome produisant `predictions.csv`.
- `final_model.pth` : Poids du modèle final entraîné à inclure dans le rendu.
- `model.py` : Définition de l'architecture ResNet18 avec tête adaptée.
- `lancer_experiences_globales.py` : Exécute les 9 configurations expérimentales (Baseline CNN, ResNet18, YOLO Crop x 3 sources de données).
- `GUIDE_UTILISATION.md` : Guide pas à pas de chaque script du projet.
- `SOURCES.md` : Liste des documentations et références communautaires utilisées.
