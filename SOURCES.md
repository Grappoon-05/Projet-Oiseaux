# Sources & Références du Projet — Reconnaissance d'Oiseaux

Ce document rassemble l'ensemble des sources documentaires, tutoriels officiels et fils de discussions communautaires (forums, GitHub) utilisés pour structurer et adapter le code du projet.

---

## 1. Transfer Learning & Architecture ResNet18

Pour adapter un modèle pré-entraîné sur ImageNet à nos 21 classes d'oiseaux (remplacement de la dernière couche `model.fc`) :

- **Tutoriel officiel PyTorch Transfer Learning :**  
  [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)  
  *Utilisé pour la structure du Transfer Learning, le remplacement de `model.fc` et le choix des hyperparamètres d'entraînement.*

- **Forum PyTorch Discuss — Modification de la couche FC :**  
  [https://discuss.pytorch.org/t/how-to-modify-the-final-fc-layer-based-on-the-torch-model/766](https://discuss.pytorch.org/t/how-to-modify-the-final-fc-layer-based-on-the-torch-model/766)  
  *Explication sur l'extraction de `in_features` et la réinitialisation de `nn.Linear(in_features, num_classes)`.*

- **Documentation officielle des modèles Torchvision (Weights & Transforms) :**  
  [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)  
  *Utilisé pour appliquer directement les transformations de prétraitement recommandées (`models.ResNet18_Weights.DEFAULT.transforms()`).*

---

## 2. Détection & Recadrage d'Objets avec YOLO (Étape 3)

Pour détecter la classe `bird` (classe 14 du jeu COCO) et découper automatiquement les boîtes englobantes :

- **Discussion GitHub Ultralytics — Recadrage d'objets via boîtes englobantes (Issue #2422) :**  
  [https://github.com/ultralytics/ultralytics/issues/2422](https://github.com/ultralytics/ultralytics/issues/2422)  
  *Technique pour extraire les coordonnées `xyxy` de la boîte la plus confiante et découper la région d'intérêt avec Pillow.*

- **Documentation officielle Ultralytics YOLO — Mode Predict :**  
  [https://docs.ultralytics.com/modes/predict/](https://docs.ultralytics.com/modes/predict/)  
  *Paramétrage des filtres de classes (`classes=[14]`), du seuil de confiance (`conf=0.25`) et du mode silencieux (`verbose=False`).*

---

## 3. Gestion des Données (Dataset, DataLoader & Split)

Pour la création des datasets et la gestion des flux d'entraînement et d'évaluation :

- **Tutoriel officiel PyTorch — Datasets & DataLoaders :**  
  [https://pytorch.org/tutorials/beginner/basics/data_tutorial.html](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)  
  *Structure des classes héritant de `torch.utils.data.Dataset` avec les méthodes `__len__` et `__getitem__`.*

- **Forum PyTorch Discuss — Séparation des dossiers Train et Test :**  
  [https://discuss.pytorch.org/t/how-to-use-imagefolder-with-a-generator-or-train-test-split/35227](https://discuss.pytorch.org/t/how-to-use-imagefolder-with-a-generator-or-train-test-split/35227)  
  *Discussion expliquant pourquoi une séparation physique des dossiers (`train/`, `val/`, `test/`) est plus rigoureuse qu'un découpage aléatoire à chaque lancement pour éviter les fuites de données.*

---

## 4. Calcul des Probabilités et Sorties de Classification

Pour formater la confiance de prédiction demandée dans le fichier `predictions.csv` :

- **Forum PyTorch Discuss — Probabilités avec Softmax et Argmax :**  
  [https://discuss.pytorch.org/t/how-to-get-probability-of-prediction/124231](https://discuss.pytorch.org/t/how-to-get-probability-of-prediction/124231)  
  *Transformation des logits bruts du modèle en score de confiance compris entre 0 et 1 via `torch.softmax(logits, dim=1)`.*
