import json
import random
import shutil
from pathlib import Path
import pandas as pd

# Emplacements demandés
PROJEX_DIR = Path("/Users/grappoon/Desktop/IUT/L2/S3/R3.14 - IA/projex")
DATA_HACKATON_SRC = PROJEX_DIR / "Oiseaux_data" / "Data_Hackaton"
DATA_LAURENS_SRC = PROJEX_DIR / "Oiseaux_data" / "Oiseaux_Laurens"
CSV_ANNOTATIONS = PROJEX_DIR / "dataset_annote.csv"

# Dossiers de sortie dans projex
HACKATON_SPLIT = PROJEX_DIR / "hackaton_split"
LAURENS_SPLIT = PROJEX_DIR / "laurens_split"
COMBINED_SPLIT = PROJEX_DIR / "combined_split"

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_target_classes():
    with open("classes.json", "r", encoding="utf-8") as f:
        raw = json.load(f)["classes"]
    classes = []
    for c in raw:
        if c not in ("autre_oiseau", "incertain"):
            classes.append(c.strip().replace(" ", "_"))
    return sorted(list(set(classes)))


def preparer_hackaton():
    print("--- 1. Découpage Hackaton (70% Train, 15% Val, 15% Test) dans projex ---")
    if HACKATON_SPLIT.exists():
        shutil.rmtree(HACKATON_SPLIT)

    for sub in ["train", "val", "test"]:
        (HACKATON_SPLIT / sub).mkdir(parents=True, exist_ok=True)

    classes = get_target_classes()
    random.seed(42)

    total_train, total_val, total_test = 0, 0, 0

    for cls in classes:
        cls_dir = DATA_HACKATON_SRC / cls
        if not cls_dir.is_dir():
            continue

        images = [f for f in sorted(cls_dir.iterdir()) if f.suffix.lower() in EXTS]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        imgs_train = images[:n_train]
        imgs_val = images[n_train:n_train + n_val]
        imgs_test = images[n_train + n_val:]

        for sub, img_list in [("train", imgs_train), ("val", imgs_val), ("test", imgs_test)]:
            dest_dir = HACKATON_SPLIT / sub / cls
            dest_dir.mkdir(exist_ok=True)
            for img in img_list:
                shutil.copy2(img, dest_dir / img.name)

        total_train += len(imgs_train)
        total_val += len(imgs_val)
        total_test += len(imgs_test)

    print(f"Hackaton créé : Train={total_train} | Val={total_val} | Test={total_test}")


def preparer_laurens():
    print("\n--- 2. Découpage Laurens par JOUR (Règle R2) dans projex ---")
    if LAURENS_SPLIT.exists():
        shutil.rmtree(LAURENS_SPLIT)

    for sub in ["train", "val", "test"]:
        (LAURENS_SPLIT / sub).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_ANNOTATIONS)
    df = df[df["label"].str.lower() != "incertain"].copy()

    # Normalisation : ex: erithacus_rubecula -> Erithacus_rubecula / rien -> rien
    def norm_label(lbl):
        l = str(lbl).strip().lower()
        if l == "rien":
            return "rien"
        parts = l.split("_")
        if len(parts) >= 2:
            return parts[0].capitalize() + "_" + parts[1].lower()
        return l.capitalize()

    df["label_norm"] = df["label"].apply(norm_label)
    df["date"] = pd.to_datetime(df["exif_datetime"]).dt.date

    jours = sorted(df["date"].unique())
    # 11 jours train, 2 jours val, 3 jours test
    jours_train = set(jours[:11])
    jours_val = set(jours[11:13])
    jours_test = set(jours[13:])

    comptes = {"train": 0, "val": 0, "test": 0}

    for _, row in df.iterrows():
        d = row["date"]
        if d in jours_train:
            sub = "train"
        elif d in jours_val:
            sub = "val"
        else:
            sub = "test"

        rel = row["image_path"].replace("\\", "/").split("Oiseaux_Laurens/")[-1]
        src_img = DATA_LAURENS_SRC / rel
        if not src_img.exists():
            continue

        lbl = row["label_norm"]
        dest = LAURENS_SPLIT / sub / lbl
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dest / src_img.name)
        comptes[sub] += 1

    print(f"Laurens créé : Train={comptes['train']} | Val={comptes['val']} | Test={comptes['test']}")


def preparer_combine():
    print("\n--- 3. Création du dataset COMBINÉ dans projex ---")
    if COMBINED_SPLIT.exists():
        shutil.rmtree(COMBINED_SPLIT)

    for sub in ["train", "val", "test"]:
        dest_sub = COMBINED_SPLIT / sub
        dest_sub.mkdir(parents=True, exist_ok=True)

        # Copie Hackaton
        src_h = HACKATON_SPLIT / sub
        if src_h.exists():
            for c_dir in src_h.iterdir():
                if c_dir.is_dir():
                    (dest_sub / c_dir.name).mkdir(exist_ok=True)
                    for f in c_dir.iterdir():
                        if f.suffix.lower() in EXTS:
                            shutil.copy2(f, dest_sub / c_dir.name / f"hack_{f.name}")

        # Copie Laurens
        src_l = LAURENS_SPLIT / sub
        if src_l.exists():
            for c_dir in src_l.iterdir():
                if c_dir.is_dir():
                    (dest_sub / c_dir.name).mkdir(exist_ok=True)
                    for f in c_dir.iterdir():
                        if f.suffix.lower() in EXTS:
                            shutil.copy2(f, dest_sub / c_dir.name / f"lau_{f.name}")

    print("Combiné créé avec succès dans :", COMBINED_SPLIT)


if __name__ == "__main__":
    preparer_hackaton()
    preparer_laurens()
    preparer_combine()
