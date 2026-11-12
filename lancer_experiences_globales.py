"""Orchestrateur central des expériences du Projet Reconnaissance d'Oiseaux.

Ce script :
1. Utilise config.json pour paramétrer chaque expérience.
2. Appelle les scripts officiels du projet (CNN_base.py, train_resnet.py, test_model.py, test_resnet.py, yolo_crop.py).
3. Génère et archive automatiquement pour CHAQUE modèle (issu de chaque dataset) et CHAQUE test :
   - La courbe de loss d'entraînement
   - La matrice de confusion sur le test Hackaton
   - La matrice de confusion sur le test Laurens (piège photo)
   - Le rapport textuel complet par espèce (Precision, Recall, F1-Score)
4. Compile tout dans un dossier prêt pour le rapport : `dossier_rapport/`
5. Met à jour le tableau récapitulatif `TABLEAU_DES_GAINS.csv`.
"""

import copy
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Chemins
CONFIG_PATH = Path("config.json")
PROJEX_DIR = Path("/Users/grappoon/Desktop/IUT/L2/S3/R3.14 - IA/projex")
OUTPUT_CSV = Path("TABLEAU_DES_GAINS.csv")
OUTPUT_CSV_PROJEX = PROJEX_DIR / "TABLEAU_DES_GAINS.csv"

# Dossier d'exportation pour le rapport
DOSSIER_RAPPORT = Path("dossier_rapport")
DOSSIER_RAPPORT.mkdir(parents=True, exist_ok=True)

# Sources disponibles dans projex
DATASET_SOURCES = {
    "hackaton": {
        "train": str(PROJEX_DIR / "hackaton_split" / "train"),
        "val": str(PROJEX_DIR / "hackaton_split" / "val"),
        "test": str(PROJEX_DIR / "hackaton_split" / "test"),
    },
    "laurens": {
        "train": str(PROJEX_DIR / "laurens_split" / "train"),
        "val": str(PROJEX_DIR / "laurens_split" / "val"),
        "test": str(PROJEX_DIR / "laurens_split" / "test"),
    },
    "combine": {
        "train": str(PROJEX_DIR / "combined_split" / "train"),
        "val": str(PROJEX_DIR / "combined_split" / "val"),
        "test": str(PROJEX_DIR / "combined_split" / "test"),
    },
}


def lire_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def ecrire_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def lancer_script(nom_script):
    """Exécute un script python avec affichage en direct et retourne sa sortie textuelle."""
    cmd = [sys.executable, "-u", nom_script]
    print(f"\n>>> Exécution : {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    lignes_sortie = []
    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)
        lignes_sortie.append(line)
        
    process.stdout.close()
    returncode = process.wait()
    
    sortie_totale = "".join(lignes_sortie)
    if returncode != 0:
        raise RuntimeError(f"Échec de {nom_script} avec code {returncode}")
        
    return sortie_totale


def extraire_metriques(sortie_test):
    """Extrait accuracy et perte depuis la sortie de test."""
    loss = None
    acc = None
    for ligne in sortie_test.splitlines():
        if "Perte moyenne" in ligne and ":" in ligne:
            try:
                loss = float(ligne.split(":")[-1].strip())
            except ValueError:
                pass
        if "Précision globale" in ligne and ":" in ligne:
            try:
                partie = ligne.split(":")[-1].strip().split("%")[0].strip()
                acc = float(partie)
            except ValueError:
                pass
    return acc, loss


def executer_experience(etape, nom_source, script_train, script_test, weights_name, loss_plot_name, img_size=None, epochs=None, lr=None):
    print("\n" + "=" * 65)
    print(f"EXPÉRIENCE : {etape} | Source : {nom_source}")
    print("=" * 65)

    nom_prefixe = f"{etape}_{nom_source}"
    cfg = lire_config()

    train_dir = DATASET_SOURCES[nom_source]["train"]
    val_dir = DATASET_SOURCES[nom_source]["val"]
    test_dir_hack = DATASET_SOURCES["hackaton"]["test"]
    test_dir_lau = DATASET_SOURCES["laurens"]["test"]

    # 1. Mise à jour de config.json pour l'entraînement
    cfg["data"]["train_dir"] = train_dir
    cfg["data"]["val_dir"] = val_dir
    
    # Utilisation des paramètres passés en priorité, sinon lecture directe de config.json
    if img_size is not None:
        cfg["data"]["img_size"] = [img_size, img_size]
    if epochs is not None:
        cfg["training"]["epochs"] = epochs
    if lr is not None:
        cfg["training"]["learning_rate"] = lr

    cfg["output"]["weights_path"] = weights_name
    cfg["output"]["loss_plot_path"] = loss_plot_name

    ecrire_config(cfg)

    # 2. Entraînement du modèle
    lancer_script(script_train)

    # Copie de la courbe de perte dans le dossier rapport
    fichier_courbe = Path(loss_plot_name)
    if fichier_courbe.exists():
        dest_courbe = DOSSIER_RAPPORT / f"courbe_loss_{nom_prefixe}.png"
        shutil.copy2(fichier_courbe, dest_courbe)

    # 3. Test sur le jeu Hackaton
    cfg["data"]["test_dir"] = test_dir_hack
    dest_mat_hack = DOSSIER_RAPPORT / f"matrice_confusion_hackaton_{nom_prefixe}.png"
    cfg["output"]["confusion_matrix_path"] = str(dest_mat_hack)
    ecrire_config(cfg)
    sortie_hack = lancer_script(script_test)
    acc_hack, loss_hack = extraire_metriques(sortie_hack)

    # Sauvegarde du rapport texte Hackaton
    (DOSSIER_RAPPORT / f"rapport_test_hackaton_{nom_prefixe}.txt").write_text(sortie_hack, encoding="utf-8")

    # Si le script de test a sauvegardé sous le nom par défaut à la racine, on s'assure qu'il est déplacé
    for nom_mat in ["matrice_confusion_baseline.png", "matrice_confusion_resnet.png"]:
        p = Path(nom_mat)
        if p.exists():
            shutil.copy2(p, dest_mat_hack)
            p.unlink()

    # 4. Test sur le jeu Laurens (piège photo)
    cfg["data"]["test_dir"] = test_dir_lau
    dest_mat_lau = DOSSIER_RAPPORT / f"matrice_confusion_laurens_{nom_prefixe}.png"
    cfg["output"]["confusion_matrix_path"] = str(dest_mat_lau)
    ecrire_config(cfg)
    sortie_lau = lancer_script(script_test)
    acc_lau, loss_lau = extraire_metriques(sortie_lau)

    # Sauvegarde du rapport texte Laurens
    (DOSSIER_RAPPORT / f"rapport_test_laurens_{nom_prefixe}.txt").write_text(sortie_lau, encoding="utf-8")

    # Si le script de test a sauvegardé sous le nom par défaut à la racine, on s'assure qu'il est déplacé
    for nom_mat in ["matrice_confusion_baseline.png", "matrice_confusion_resnet.png"]:
        p = Path(nom_mat)
        if p.exists():
            shutil.copy2(p, dest_mat_lau)
            p.unlink()

    # Copie des prédictions d'exemples
    for nom_pred in ["test_predictions.png", "test_predictions_resnet.png"]:
        p = Path(nom_pred)
        if p.exists():
            shutil.copy2(p, DOSSIER_RAPPORT / f"exemples_predictions_{nom_prefixe}.png")
            break

    print(f"\n--> Synthèse {etape} ({nom_source}) :")
    print(f"    Test Hackaton : {acc_hack}% (Loss {loss_hack})")
    print(f"    Test Laurens  : {acc_lau}% (Loss {loss_lau})")

    return {
        "etape": etape,
        "source_train": nom_source,
        "acc_hackaton_test": acc_hack,
        "loss_hackaton_test": loss_hack,
        "acc_laurens_test": acc_lau,
        "loss_laurens_test": loss_lau,
        "courbe_loss": f"dossier_rapport/courbe_loss_{nom_prefixe}.png",
        "matrice_conf_hackaton": f"dossier_rapport/matrice_confusion_hackaton_{nom_prefixe}.png",
        "matrice_conf_laurens": f"dossier_rapport/matrice_confusion_laurens_{nom_prefixe}.png",
        "poids_sauvegardes": weights_name,
    }


def main():
    config_initiale = copy.deepcopy(lire_config())
    resultats = []

    # Lecture des paramètres depuis config.json
    cfg_train = config_initiale.get("training", {})
    epochs_cfg = cfg_train.get("epochs", 10)
    lr_cfg = cfg_train.get("learning_rate", 0.001)

    print("\n" + "=" * 65)
    print(f"CONFIGURATION GLOBALE DÉTECTÉE DEPUIS config.json :")
    print(f" - Nombre d'époques (epochs) : {epochs_cfg}")
    print(f" - Taux d'apprentissage (lr)  : {lr_cfg}")
    print("=" * 65)

    try:
        # ====================================================
        # ÉTAPE 1 : BASELINE CNN (CNN_base.py & test_model.py)
        # ====================================================
        for src in ["hackaton", "laurens", "combine"]:
            res = executer_experience(
                etape="Etape1_Baseline",
                nom_source=src,
                script_train="CNN_base.py",
                script_test="test_model.py",
                weights_name=f"baseline_{src}.pth",
                loss_plot_name=f"loss_baseline_{src}.png",
                img_size=128,
                epochs=epochs_cfg,
                lr=lr_cfg,
            )
            resultats.append(res)

        # ====================================================
        # ÉTAPE 2 : TRANSFER LEARNING (train_resnet.py & test_resnet.py)
        # ====================================================
        for src in ["hackaton", "laurens", "combine"]:
            res = executer_experience(
                etape="Etape2_ResNet18",
                nom_source=src,
                script_train="train_resnet.py",
                script_test="test_resnet.py",
                weights_name=f"resnet18_{src}.pth",
                loss_plot_name=f"loss_resnet18_{src}.png",
                img_size=224,
                epochs=epochs_cfg,
                lr=lr_cfg,
            )
            resultats.append(res)

        # ====================================================
        # ÉTAPE 3 : YOLO CROP + RESNET18
        # ====================================================
        for src in ["hackaton", "laurens", "combine"]:
            res = executer_experience(
                etape="Etape3_YOLO_ResNet",
                nom_source=src,
                script_train="train_resnet.py",
                script_test="test_resnet.py",
                weights_name=f"resnet18_yolo_{src}.pth",
                loss_plot_name=f"loss_resnet18_yolo_{src}.png",
                img_size=224,
                epochs=epochs_cfg,
                lr=lr_cfg,
            )
            resultats.append(res)

        # Sauvegarde du tableau complet
        champs = [
            "etape",
            "source_train",
            "acc_hackaton_test",
            "loss_hackaton_test",
            "acc_laurens_test",
            "loss_laurens_test",
            "courbe_loss",
            "matrice_conf_hackaton",
            "matrice_conf_laurens",
            "poids_sauvegardes",
        ]
        for dest in [OUTPUT_CSV, OUTPUT_CSV_PROJEX, DOSSIER_RAPPORT / "TABLEAU_DES_GAINS.csv"]:
            try:
                with open(dest, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=champs)
                    writer.writeheader()
                    writer.writerows(resultats)
                print(f"Tableau récapitulatif sauvegardé dans : {dest}")
            except Exception as e:
                print(f"Erreur lors de l'enregistrement de {dest} : {e}")

        print("\n" + "=" * 65)
        print("TOUTES LES EXPÉRIENCES SONT TERMINÉES ET ARCHIVÉES !")
        print(f"L'intégralité des fichiers pour votre rapport se trouve dans :")
        print(f"👉 {DOSSIER_RAPPORT.resolve()}")
        print("=" * 65)

    finally:
        # Restauration de config.json
        ecrire_config(config_initiale)
        print("\nconfig.json restauré dans son état initial.")


if __name__ == "__main__":
    main()
