"""Launcher central des expériences du Projet Reconnaissance d'Oiseaux.

Ce script agit comme un chef d'orchestre :
- Il met à jour config.json pour chaque scénario expérimental.
- Il appelle directement les scripts officiels du projet (CNN_base.py, train_resnet.py, test_model.py, test_resnet.py, yolo_crop.py).
- Il compile l'ensemble des scores dans TABLEAU_DES_GAINS.csv.
"""

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

# Chemins
CONFIG_PATH = Path("config.json")
PROJEX_DIR = Path("/Users/grappoon/Desktop/IUT/L2/S3/R3.14 - IA/projex")
OUTPUT_CSV = Path("TABLEAU_DES_GAINS.csv")
OUTPUT_CSV_PROJEX = PROJEX_DIR / "TABLEAU_DES_GAINS.csv"

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
    """Exécute un script python et retourne sa sortie textuelle."""
    cmd = [sys.executable, nom_script]
    print(f"\n>>> Exécution : {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Erreur lors de l'exécution de {nom_script} :")
        print(result.stderr)
        raise RuntimeError(f"Échec de {nom_script}")
    print(result.stdout)
    return result.stdout


def extraire_metriques(sortie_test):
    """Extrait accuracy et perte depuis la sortie standard de test_model.py ou test_resnet.py."""
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


def executer_experience(etape, nom_source, script_train, script_test, img_size, epochs, lr, weights_name, loss_plot_name, crop_source_dir=None):
    print("\n" + "=" * 60)
    print(f"EXPÉRIENCE : {etape} | Source : {nom_source}")
    print("=" * 60)

    cfg = lire_config()

    # Si c'est l'étape YOLO, on lance le recadrage si précisé
    train_dir = DATASET_SOURCES[nom_source]["train"]
    val_dir = DATASET_SOURCES[nom_source]["val"]
    test_dir_hack = DATASET_SOURCES["hackaton"]["test"]
    test_dir_lau = DATASET_SOURCES["laurens"]["test"]

    # Mise à jour de config.json pour l'entraînement
    cfg["data"]["train_dir"] = train_dir
    cfg["data"]["val_dir"] = val_dir
    cfg["data"]["img_size"] = [img_size, img_size]
    cfg["training"]["epochs"] = epochs
    cfg["training"]["learning_rate"] = lr
    cfg["output"]["weights_path"] = weights_name
    cfg["output"]["loss_plot_path"] = loss_plot_name

    ecrire_config(cfg)
    print(f"config.json mis à jour pour l'entraînement de {etape}")

    # 1. Entraînement via le script dédié
    lancer_script(script_train)

    # 2. Test sur Hackaton
    cfg["data"]["test_dir"] = test_dir_hack
    ecrire_config(cfg)
    sortie_hack = lancer_script(script_test)
    acc_hack, loss_hack = extraire_metriques(sortie_hack)

    # 3. Test sur Laurens (piège photo)
    cfg["data"]["test_dir"] = test_dir_lau
    ecrire_config(cfg)
    sortie_lau = lancer_script(script_test)
    acc_lau, loss_lau = extraire_metriques(sortie_lau)

    print(f"--> Résultat {etape} ({nom_source}) :")
    print(f"    Test Hackaton : {acc_hack}% (Loss {loss_hack})")
    print(f"    Test Laurens  : {acc_lau}% (Loss {loss_lau})")

    return {
        "etape": etape,
        "source_train": nom_source,
        "acc_hackaton_test": acc_hack,
        "loss_hackaton_test": loss_hack,
        "acc_laurens_test": acc_lau,
        "loss_laurens_test": loss_lau,
        "poids_sauvegardes": weights_name
    }


def main():
    config_initiale = copy.deepcopy(lire_config())
    resultats = []

    try:
        # ----------------------------------------------------
        # ÉTAPE 1 : CNN BASELINE (CNN_base.py & test_model.py)
        # ----------------------------------------------------
        for src in ["hackaton", "laurens", "combine"]:
            res = executer_experience(
                etape="Etape1_Baseline",
                nom_source=src,
                script_train="CNN_base.py",
                script_test="test_model.py",
                img_size=128,
                epochs=10,
                lr=0.001,
                weights_name=f"baseline_{src}.pth",
                loss_plot_name=f"loss_baseline_{src}.png"
            )
            resultats.append(res)

        # ----------------------------------------------------
        # ÉTAPE 2 : TRANSFER LEARNING (train_resnet.py & test_resnet.py)
        # ----------------------------------------------------
        for src in ["hackaton", "laurens", "combine"]:
            res = executer_experience(
                etape="Etape2_ResNet18",
                nom_source=src,
                script_train="train_resnet.py",
                script_test="test_resnet.py",
                img_size=224,
                epochs=10,
                lr=0.0005,
                weights_name=f"resnet18_{src}.pth",
                loss_plot_name=f"loss_resnet18_{src}.png"
            )
            resultats.append(res)

        # ----------------------------------------------------
        # ÉTAPE 3 : YOLO CROP + RESNET18
        # On s'assure d'abord que yolo_crop.py est exécuté
        # ----------------------------------------------------
        print("\n" + "=" * 60)
        print("ÉTAPE 3 : Recadrage préliminaire avec yolo_crop.py...")
        print("=" * 60)
        try:
            lancer_script("yolo_crop.py")
        except Exception as e:
            print(f"Avertissement lors du recadrage YOLO : {e}")

        for src in ["hackaton", "laurens", "combine"]:
            res = executer_experience(
                etape="Etape3_YOLO_ResNet",
                nom_source=src,
                script_train="train_resnet.py",
                script_test="test_resnet.py",
                img_size=224,
                epochs=10,
                lr=0.0005,
                weights_name=f"resnet18_yolo_{src}.pth",
                loss_plot_name=f"loss_resnet18_yolo_{src}.png"
            )
            resultats.append(res)

        # Sauvegarde du tableau final
        champs = ["etape", "source_train", "acc_hackaton_test", "loss_hackaton_test", "acc_laurens_test", "loss_laurens_test", "poids_sauvegardes"]
        for dest in [OUTPUT_CSV, OUTPUT_CSV_PROJEX]:
            try:
                with open(dest, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=champs)
                    writer.writeheader()
                    writer.writerows(resultats)
                print(f"Tableau récapitulatif sauvegardé dans : {dest}")
            except Exception as e:
                print(f"Erreur d'écriture sur {dest} : {e}")

    finally:
        # Restauration propre de config.json
        ecrire_config(config_initiale)
        print("\nconfig.json restauré dans son état initial.")


if __name__ == "__main__":
    main()
