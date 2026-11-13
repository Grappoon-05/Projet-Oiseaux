# Administration Log — Projet Reconnaissance d'Oiseaux

## 2026-11-12 — Clôture des Expériences Globales et Génération du Compte-Rendu Final

### Statut
- **Version du Projet :** `v1.2.0` (Validation complète des 9 runs expérimentaux)
- **Objectifs atteints :** 100 % des étapes 1, 2 et 3 exécutées, évaluées et documentées.

### Actions Réalisées
1. **Exécution des 9 configurations expérimentales** via `lancer_experiences_globales.py` :
   - Étape 1 : Baseline CNN sur `hackaton`, `laurens`, `combine`.
   - Étape 2 : Transfer Learning ResNet18 sur `hackaton`, `laurens`, `combine`.
   - Étape 3 : YOLO Crop (yolo11n.pt) + ResNet18 sur `hackaton`, `laurens`, `combine`.
2. **Archivage systématique dans `dossier_rapport/`** :
   - 9 courbes d'apprentissage de perte (`courbe_loss_*.png`).
   - 18 matrices de confusion complètes (`matrice_confusion_hackaton_*.png` et `matrice_confusion_laurens_*.png`).
   - 18 rapports textuels de métriques par classe (Precision, Recall, F1-Score).
   - Enregistrement des poids de chaque modèle (`.pth`).
3. **Consolidation du Tableau des Gains :**
   - Mise à jour de `TABLEAU_DES_GAINS.csv` à la racine et dans `dossier_rapport/`.
4. **Sélection et préparation du modèle final :**
   - Copie du meilleur modèle retenu (`resnet18_yolo_combine.pth`) vers `final_model.pth` pour utilisation directe avec `predict.py`.
5. **Rédaction du Compte-Rendu complet :**
   - Mise à jour détaillée de `RAPPORT_SYNTHESE.md` avec l'analyse du décrochage de domaine, l'impact du Transfer Learning et la valeur ajoutée du recadrage YOLO.
6. **Livrables complémentaires :**
   - Notebook interactif `lancer_experiences_globales.ipynb` prêt pour Google Colab ou machine locale.
   - Pack autonome `pack_gpu/` avec `requirements.txt` adapté CUDA 12.1.
