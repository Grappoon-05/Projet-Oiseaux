# Directives et Feuille de Route — Projet Reconnaissance d'Oiseaux

Ce document synthétise les consignes du sujet (`Sujet_projet_oiseaux.pdf`), les règles impératives, la démarche expérimentale attendue, ainsi que la liste des tâches à accomplir pour le binôme.

---

## 1. Objectifs du Projet

Développer une chaîne de traitement complète capable d'identifier les espèces d'oiseaux visitant une mangeoire de jardin à partir des images brutes d'un piège photo (`Oiseaux_Laurens/`).

### Contraintes clés du problème :
- Les images d'entraînement annotées initiales proviennent d'un jeu différent : `Data_Hackaton/` (conditions de prise de vue très variées, beaucoup d'espèces hors périmètre).
- Les images du piège photo `Oiseaux_Laurens/` **ne sont pas annotées**.
- Il y a un fort décalage de domaine (*domain shift*) entre les deux jeux de données.
- Beaucoup d'images du piège photo ne contiennent aucun oiseau (déclenchements à vide, vent, rongeurs, etc.).

---

## 2. Règles Impératives du Sujet

| Règle | Description |
| :--- | :--- |
| **R1 (Pas de fuite)** | Tout résultat chiffré doit être mesuré sur des données que le modèle **n'a jamais vues** lors de son entraînement. |
| **R2 (Split par JOUR)** | **Découper les données par JOURNÉE ENTIÈRE, jamais par fichier ou au hasard.** Deux images prises à quelques secondes d'intervalle lors d'un même passage d'oiseau sont quasi-identiques. Mettre l'une en train et l'autre en val fausserait complètement l'évaluation. |
| **R3 (Mesure des gains)** | Chaque technique ajoutée doit obligatoirement être accompagnée d'un gain chiffré dans le tableau de bord des expériences. |
| **R4 (Reproductibilité)** | Scripts exécutables sans interaction humaine (ni `input()`, ni déroulement manuel de notebook). Chemins relatifs ou passés en arguments CLI. |
| **RGPD / Confidentialité** | Des personnes apparaissent sur certaines photos. Les dossiers `Data_Hackaton/`, `Oiseaux_Laurens/` et les fichiers de poids `*.pt` **ne doivent jamais être commités ou mis en ligne publiquement** (vérifier `.gitignore`). |

---

## 3. Spécifications Techniques des Livrables

### 3.1. Commandes du README (Annexe A)
Le projet doit proposer deux scripts autonomes exécutables depuis la racine :
1. **Entraînement :**
   ```bash
   python train.py <racine_des_donnees> <dossier_de_sortie>
   ```
2. **Inférence :**
   ```bash
   python predict.py <dossier_images> <dossier_de_sortie>
   ```

### 3.2. Format du CSV de Prédictions (`predictions.csv`)
En-tête exact, sans index inutile :
```csv
image_id,sequence_id,label,confidence
```
- `image_id` : Nom du fichier sans chemin ni extension (ex: `IMG_0042`).
- `sequence_id` : Identifiant du passage temporel groupé (étape 5).
- `label` : Un des labels déclarés dans `classes.json`.
- `confidence` : Score de confiance entre 0.0 et 1.0.

### 3.3. Périmètre des Classes (`classes.json`)
- Noms latins obligatoires selon la nomenclature de l'Annexe B (ex: `Parus_major`, `Cyanistes_caeruleus`, `Erithacus_rubecula`...). **Attention aux underscores `_`**.
- Deux étiquettes obligatoires :
  - `"rien"` : Déclenchement à vide, mammifère, humain, nuit noire.
  - `"autre_oiseau"` : Espèce d'oiseau non incluse dans vos classes définies.
- *Interdiction formelle d'avoir `"incertain"` en sortie de modèle.*

---

## 4. Étapes Méthodologiques à Valider

```
[Étape 0: Exploration & Découpage EXIF]
                  │
                  ▼
[Étape 1: Baseline CNN From Scratch]
                  │
                  ▼
[Étape 2: Transfer Learning (ResNet18 / ImageNet)]
                  │
                  ▼
[Étape 3: Détection YOLO (Crop 'bird') avant Classification]
                  │
                  ▼
[Étape 4: Comparaison des Sources (Hackaton vs Mangeoire vs Hybride)]
                  │
                  ▼
[Étape 5: Décision par Séquence/Visite (Agrégation temporelle)]
```

### Étape 0 : Exploration, Nettoyage et Annotation
- Analyser la distribution des classes, les résolutions et les métadonnées EXIF.
- Répartir les jours d'observation : réserver des journées entières de piège photo pour le test final.
- **Annoter manuellement** un lot de séquences de `Oiseaux_Laurens/` (avec la date EXIF pour annoter par rafale).
- Corriger et figer `classes.json`.

### Étape 1 : Baseline honnête
- Entraîner un petit CNN classique sur les images entières de `Data_Hackaton/`.
- Évaluer deux fois :
  1. Sur le set de test `Data_Hackaton/`.
  2. Sur le sous-ensemble annoté de `Oiseaux_Laurens/`.
- Noter l'écart de performance (mesure du décalage de domaine).

### Étape 2 : Transfer Learning
- Utiliser un modèle pré-entraîné (ex: `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`).
- Adapter la dernière couche linéaire (`fc`) au nombre de classes de `classes.json`.
- Normaliser selon ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- Mesurer le gain par rapport à l'étape 1.

### Étape 3 : Détecter avant de classer (YOLO Crop)
- Utiliser `ultralytics` avec `yolo11s.pt` (classe COCO 14 : `bird`).
- Rogner l'image sur l'oiseau détecté avant de l'envoyer au classifieur.
- Si aucun oiseau n'est détecté sur le piège photo, assigner directement la classe `"rien"`.
- Optimiser le seuil de confiance de détection par rapport à l'accuracy finale.

### Étape 4 : Origine des données d'entraînement
Comparer 3 entraînements distincts avec le même protocole d'évaluation :
1. Uniquement sur `Data_Hackaton/`.
2. Uniquement sur les images annotées de `Oiseaux_Laurens/`.
3. Sur les deux jeux combinés.

### Étape 5 : Agrégation temporelle par visite
- Regrouper les images d'une même rafale/visite à partir des horodatages EXIF (seuil $\sim$ 60 secondes).
- Agréger les prédictions (vote majoritaire, moyenne des probabilités).
- Comparer l'accuracy au niveau image vs au niveau passage (`sequence_id`).

---

## 5. Livrables Finaux et Barème

- **Rendu (70%)** :
  - **Rapport de 3 à 6 pages** : démarche, justifications, tableau des gains, analyse d'erreurs, limites.
  - **Dépôt Git propre** : code reproductible, `requirements.txt`, `README.md`, scripts `train.py` et `predict.py`.
  - **Poids du modèle final** et `classes.json`.
- **Soutenance (30%)** :
  - Présentation de 10 minutes + questions.
  - Axée sur la justification des choix et l'analyse critique des résultats.
