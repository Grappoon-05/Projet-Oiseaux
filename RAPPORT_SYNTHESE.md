# Synthèse Complète des Expériences — Projet Reconnaissance d'Oiseaux

Ce document synthétise l'intégralité des 9 runs expérimentaux menés conformément aux exigences du sujet (Étape 1 Baseline CNN, Étape 2 ResNet18 Transfer Learning, Étape 3 YOLO Crop + ResNet18), croisés sur les 3 sources de données (Hackaton seul, Piège photo Laurens seul, et Jeu combiné).

---

## 1. Tableau Récapitulatif Officiel des Gains

| Étape | Modèle | Données d'Entraînement | Test Hackaton (Acc / Loss) | Test Laurens - Piège photo (Acc / Loss) | Fichier de Poids Sauvegardé |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Étape 1** | Baseline CNN | Hackaton seul | **18.75%** (2.65) | **0.00%** (3.36) | `Etape1_Baseline_hackaton_CNN_base.pth` |
| **Étape 1** | Baseline CNN | Laurens seul | **3.91%** (11.32) | **8.82%** (2.91) | `Etape1_Baseline_laurens_CNN_base.pth` |
| **Étape 1** | Baseline CNN | Combiné | **9.38%** (2.86) | **2.45%** (1.66) | `Etape1_Baseline_combine_CNN_base.pth` |
| **Étape 2** | ResNet18 (ImageNet) | Hackaton seul | **43.75%** (2.21) | **1.47%** (4.65) | `Etape2_ResNet18_hackaton_ResNet18.pth` |
| **Étape 2** | ResNet18 (ImageNet) | Laurens seul | **7.03%** (9.40) | **90.20%** (0.83) | `Etape2_ResNet18_laurens_ResNet18.pth` |
| **Étape 2** | ResNet18 (ImageNet) | Combiné | **42.97%** (2.27) | **62.75%** (1.06) | `Etape2_ResNet18_combine_ResNet18.pth` |
| **Étape 3** | YOLO Crop + ResNet18 | Hackaton seul | **57.81%** (1.60) | **16.67%** (2.62) | `Etape3_YOLO_ResNet_hackaton_ResNet18.pth` |
| **Étape 3** | YOLO Crop + ResNet18 | Laurens seul | **4.69%** (8.05) | **64.71%** (1.38) | `Etape3_YOLO_ResNet_laurens_ResNet18.pth` |
| **Étape 3** | **YOLO Crop + ResNet18** | **Combiné** | **45.31%** (2.32) | **88.24%** (0.44) | **`Etape3_YOLO_ResNet_combine_ResNet18.pth`** *(Modèle Final)* |

---

## 2. Analyse Critique des Résultats pour le Rapport (3 à 6 pages)

### 2.1. Mise en évidence éclatante du "Domain Shift" (Écart de domaine)
- **Modèle entraîné sur Hackaton seul :**
  - Même avec ResNet18 à 43.75% sur Hackaton, il s'effondre à **1.47%** sur le piège photo.
  - **Explication scientifique :** Les photos Hackaton sont des photos naturalistes (oiseau centré, bonne lumière, pas de perchoir fixe), alors que le piège photo a une focale fixe, un arrière-plan récurrent et des oiseaux souvent lointains ou partiellement masqués.
- **Modèle entraîné sur Laurens seul :**
  - Il atteint un excellent score sur son propre domaine (**90.20%**), mais s'effondre à **7.03%** sur Hackaton. Il a sur-appris le décor et les angles spécifiques de la mangeoire.
- **L'intérêt fondamental du jeu Combiné :**
  - L'entraînement combiné résout ce compromis : il conserve une bonne généralisation sur Hackaton (**45.31%**) tout en atteignant **88.24%** sur le piège photo !

### 2.2. Apport du Transfer Learning (Étape 2 vs Étape 1)
- Le passage du CNN *from scratch* à **ResNet18** fait bondir les scores de manière spectaculaire :
  - Sur Hackaton : passage de **18.75% à 43.75%** (+25 points).
  - Sur Laurens : passage de **8.82% à 90.20%** (+81 points).
- **Justification :** Le CNN simple souffre d'un manque de données (quelques centaines d'images ne suffisent pas pour converger). Les couches pré-entraînées sur ImageNet fournissent déjà les détecteurs de texture de plumes et de contours.

### 2.3. Apport du Recadrage YOLO (Étape 3)
- L'isolation de l'oiseau par YOLO avant de donner le crop à ResNet18 permet :
  - De passer le score Hackaton à son sommet historique : **57.81%** (gain de +14 points par rapport à l'image entière).
  - De réduire drastiquement la perte finale sur le jeu combiné à **0.44** (la plus basse de toutes les expériences).
  - D'éliminer le bruit de fond et de permettre à l'absence de détection de prédire directement la classe `"rien"`.

---

## 3. Emplacement des Fichiers et Graphiques Générés

Tous les graphiques ont été automatiquement tracés et enregistrés dans le dossier `graphiques_resultats/` :

### 3.1. Courbes de Perte Époque par Époque (Train Loss vs Test Hackaton vs Test Laurens) :
- `courbe_loss_Etape1_Baseline_hackaton_CNN_base.png`
- `courbe_loss_Etape1_Baseline_laurens_CNN_base.png`
- `courbe_loss_Etape1_Baseline_combine_CNN_base.png`
- `courbe_loss_Etape2_ResNet18_hackaton_ResNet18.png`
- `courbe_loss_Etape2_ResNet18_laurens_ResNet18.png`
- `courbe_loss_Etape2_ResNet18_combine_ResNet18.png`
- `courbe_loss_Etape3_YOLO_ResNet_hackaton_ResNet18.png`
- `courbe_loss_Etape3_YOLO_ResNet_laurens_ResNet18.png`
- `courbe_loss_Etape3_YOLO_ResNet_combine_ResNet18.png`

### 3.2. Matrices de Confusion sur le Piège Photo Laurens (Analyse d'erreurs) :
- `matrice_confusion_lau_Etape1_Baseline_hackaton_CNN_base.png`
- `matrice_confusion_lau_Etape1_Baseline_laurens_CNN_base.png`
- `matrice_confusion_lau_Etape1_Baseline_combine_CNN_base.png`
- `matrice_confusion_lau_Etape2_ResNet18_hackaton_ResNet18.png`
- `matrice_confusion_lau_Etape2_ResNet18_laurens_ResNet18.png`
- `matrice_confusion_lau_Etape2_ResNet18_combine_ResNet18.png`
- `matrice_confusion_lau_Etape3_YOLO_ResNet_hackaton_ResNet18.png`
- `matrice_confusion_lau_Etape3_YOLO_ResNet_laurens_ResNet18.png`
- `matrice_confusion_lau_Etape3_YOLO_ResNet_combine_ResNet18.png`

### 3.3. Matrices de Confusion sur le Test Hackaton :
- 9 matrices correspondantes enregistrées sous `matrice_confusion_hack_*.png`.
