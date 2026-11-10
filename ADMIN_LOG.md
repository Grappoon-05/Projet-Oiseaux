# Journal d'Administration — Branche `model` (Projet Reconnaissance d'Oiseaux)

Ce document retrace l'intégralité des actions menées, des choix techniques, des erreurs rencontrées et de leurs résolutions depuis la création de la branche `model` jusqu'à la mise en place de la Baseline (Étape 1).

---

## 1. Contexte et Objectifs de la Branche

- **Branche :** `model`
- **Objectif initial :** Réaliser l'**Étape 1 : Baseline honnête (CNN from scratch)** selon les directives de `Sujet_projet_oiseaux.pdf`.
- **Contraintes de départ :**
  - Se baser sur l'apprentissage et le style de code des travaux pratiques (TP3 vision / TP4 NLP & architecture PyTorch).
  - Traiter les images réelles de `Data_Hackaton/` en adaptant la classification aux 21 espèces cibles de `classes.json`.
  - Assurer la reproductibilité stricte (pas d'`input()`, isolation des seeds, découpage sans fuite de données).

---

## 2. Chronologie Complète des Actions et Modifications

### Phase 1 : Cadrage du Projet et Feuilles de Route
1. **Lecture et analyse du sujet (`Sujet_projet_oiseaux.pdf`)** :
   - Identification des 6 étapes (Étape 0 à Étape 5), des contraintes RGPD (personnes sur les pièges photos), des règles d'évaluation (split par jour pour éviter les fuites de rafale).
2. **Rédaction de `DIRECTIVES.md`** :
   - Fiche technique récapitulative commitée et poussée sur `main` (`a0ed8cb`).

### Phase 2 : Résolution des Problèmes d'Environnement Python / Pip
- **Problème 1 :** Détection de plusieurs versions de Python (`3.9`, `3.11`, `3.14`).
  - *Symptôme :* `pip install` signalait que les paquets étaient déjà présents, mais sous Python 3.14 alors que 3.11 était souhaité.
  - *Investigation :* PyTorch 2.15 était déjà présent dans `~/.local/bin/python3.11` (fourni via `uv`), tandis que le binaire système de `/Library/Frameworks/` ne l'avait pas.
- **Problème 2 :** Erreur `externally-managed-environment` (PEP 668).
  - *Erreur :* `This environment is externally managed. managed by uv`.
  - *Résolution :* Ajout de `break-system-packages = true` dans `~/.config/pip/pip.conf` ou appel direct de l'interpréteur doté de PyTorch.

### Phase 3 : Conception de la Baseline CNN (`CNN_base.py`)
1. **Inspiration des TPs de cours :**
   - Étude de `TP3cnn.py`, `TP3_IA.py` et `load_dataset.py`.
   - Adaptation d'un problème de régression/colorisation CIFAR-10 vers un problème de classification multi-classes d'oiseaux.
   - Choix de la fonction de perte : `CrossEntropyLoss` (au lieu de `MSELoss` du TP3).
2. **Découplage de la configuration (`config.json`) :**
   - Création de `config.json` pour isoler les chemins, la taille d'image (`128x128`), les époques, le batch size, et les hyperparamètres de convolutions.
3. **Harmonisation des noms d'espèces :**
   - Remplacement systématique des espaces par des underscores (ex: `Parus_major`) pour correspondre exactement aux répertoires de `Data_Hackaton`.

### Phase 4 : Synchronisation du Device (`select_device.py`)
- **Problème :** Présence d'un `input()` interactif dans `selection_device.py` qui bloquait les scripts automatiques (violation de la règle R4 du sujet).
- **Modification sur `main` :** Le fichier a été corrigé et renommé en `select_device.py`.
- **Résolution dans la branche `model` :** Récupération via `git checkout origin/main -- select_device.py` et suppression de l'ancien `selection_device.py`.

### Phase 5 : Création de la Séparation Physique du Dataset (`split_dataset.py`)
- **Motivation :** L'utilisateur a demandé d'éviter le split purement dynamique en mémoire et de séparer physiquement les dossiers pour figer les données d'évaluation.
- **Évolution de la séparation :**
  1. *Version initiale :* 80% Train / 20% Test.
  2. *Version finale demandée :* **3 splits physiques** :
     - `dataset_split/train/` (70%) : 719 images.
     - `dataset_split/val/` (15%) : 146 images.
     - `dataset_split/test/` (15%) : 168 images.
- **Sécurité Git :** Ajout immédiat de `dataset_split/` dans `.gitignore` pour ne jamais commiter les images (conforme RGPD et sujet).

### Phase 6 : Séparation des Responsabilités (Train vs Val vs Test)
- **Clarification fonctionnelle :**
  - `CNN_base.py` a été épuré pour **ne s'occuper exclusivement que de l'entraînement** sur `dataset_split/train/`.
  - Création de `eval_model.py` : script autonome d'évaluation sur `dataset_split/val/`.
  - Création de `test_model.py` : script autonome d'évaluation finale sur le jeu sanctuaire `dataset_split/test/`.

---

## 3. Registre des Erreurs et Solutions Techniques

| # | Composant / Outil | Erreur Rencontrée | Cause Identifiée | Solution Appliquée |
| :-: | :--- | :--- | :--- | :--- |
| **1** | Pip / Python 3.11 | `externally-managed-environment` (PEP 668) | Marqueur `uv` protégeant l'environnement système. | Définition de `break-system-packages = true` dans la configuration utilisateur `pip.conf`. |
| **2** | PyTorch / Device | `input()` bloquant en mode CPU | Ancien script `selection_device.py` demandant une confirmation console. | Remplacement par `select_device.py` de `main` avec détection non-bloquante (`mps` / `cuda` / `cpu`). |
| **3** | Git checkout | `error: pathspec 'selection_device.py' did not match` | Le fichier avait été renommé en `select_device.py` sur `main`. | `git fetch origin` puis `git checkout origin/main -- select_device.py`. |
| **4** | Dataset / Noms | Décalage dans `classes.json` | Des classes comportaient des espaces (`Parus major`) au lieu d'underscores (`Parus_major`). | Fonction `get_target_classes()` nettoyant et convertissant les chaînes automatiquement. |
| **5** | Architecture CNN | Résultats initiaux trop faibles | Seulement 2 couches de convolutions sans Batch Normalization ni data augmentation. | Ajout d'un 3ème bloc convolutif (32 $\rightarrow$ 64 $\rightarrow$ 128), `BatchNorm2d`, `Dropout(0.3)`, et `RandomHorizontalFlip`. |
| **6** | Pipeline Data | Confusion entre validation et entraînement | La validation tournait à l'intérieur de `CNN_base.py`. | Isolation stricte : `CNN_base.py` (Train uniquement), `eval_model.py` (Val), `test_model.py` (Test). |

---

## 4. Historique des Commits de la Branche `model`

```
* b2f3d8b refactor(model): isolation stricte de train dans CNN_base.py et creation de eval_model.py
* 443ef37 feat(model): pipeline 3-splits (train/val/test), script split_dataset.py et optimisation CNN_base.py
* 0145583 A jout du fichier test_model.py qui test le modèle gnérer et ajoute le png des test de prediction
* 8e5de8e Changement de fichier passage de selection_device.py à select_device.py et mise à jour de CNN_base.py pour ajuster
* 1d73eb1 Ajout du premier modèle, de ces courbes de loss et mise à jour de CNN_base.py
* ddb3222 feat(model): ajout du CNN baseline base sur TP3 et fichier de configuration config.json
```

---

## 5. État Actuel et Prochaines Actions

### État de la branche :
- Code source modulaire, testé et syntaxiquement validé (`py_compile`).
- Fichiers volumineux ou générés (`dataset_split/`, `*.pth`, `*.png`, `.idea/`) exclus du suivi Git pour préserver la propreté du dépôt.
- Configuration centralisée dans `config.json`.

### Reste à faire pour finaliser l'Étape 1 :
1. Exécuter l'entraînement final : `python3 CNN_base.py`.
2. Relever les scores de test : `python3 test_model.py`.
3. Réaliser la confrontation sur un échantillon annoté du piège photo `Oiseaux_Laurens/` (mesure du domain shift).
