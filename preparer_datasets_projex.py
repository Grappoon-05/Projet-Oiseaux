import json
import shutil
from pathlib import Path
import pandas as pd

# Emplacement demandé par l'utilisateur
PROJEX_DIR = Path("/Users/grappoon/Desktop/IUT/L2/S3/R3.14 - IA/projex")
LAURENS_IMG_ROOT = PROJEX_DIR / "Oiseaux_data" / "Oiseaux_Laurens"
HACKATON_SPLIT_DIR = Path("./dataset_split")  # Déjà existant avec train/val/test

CSV_ANNOTATIONS = PROJEX_DIR / "dataset_annote.csv"
DEST_LAURENS_SPLIT = PROJEX_DIR / "laurens_split"
DEST_COMBINED_SPLIT = PROJEX_DIR / "combined_split"


def preparer_laurens_split():
    print("--- 1. Préparation du découpage de Laurens par JOUR (Règle R2) ---")
    df = pd.read_csv(CSV_ANNOTATIONS)
    
    # Exclusion des images "incertain" selon le sujet
    df = df[df["label"].str.lower() != "incertain"].copy()
    
    # Normalisation du nom de classe (ex: erithacus_rubecula -> Erithacus_rubecula / rien -> rien)
    def normaliser_label(lbl):
        l = str(lbl).strip().lower()
        if l == "rien":
            return "rien"
        parties = l.split("_")
        if len(parties) >= 2:
            return parties[0].capitalize() + "_" + parties[1].lower()
        return l.capitalize()
        
    df["label_norm"] = df["label"].apply(normaliser_label)
    df["date"] = pd.to_datetime(df["exif_datetime"]).dt.date
    
    # Liste ordonnée des dates
    jours = sorted(df["date"].unique())
    print(f"Total journées d'observation : {len(jours)}")
    
    # Règle R2 : Découpage par journée entière
    # ~70% des jours en train, ~15% en val, ~15% en test
    # 16 jours : 11 jours train, 2 jours val, 3 jours test
    jours_train = set(jours[:11])
    jours_val = set(jours[11:13])
    jours_test = set(jours[13:])
    
    print(f"Jours Train : {len(jours_train)} | Jours Val : {len(jours_val)} | Jours Test : {len(jours_test)}")
    
    if DEST_LAURENS_SPLIT.exists():
        shutil.rmtree(DEST_LAURENS_SPLIT)
        
    for subset in ["train", "val", "test"]:
        (DEST_LAURENS_SPLIT / subset).mkdir(parents=True, exist_ok=True)
        
    comptes = {"train": 0, "val": 0, "test": 0}
    
    for _, row in df.iterrows():
        date_obs = row["date"]
        if date_obs in jours_train:
            subset = "train"
        elif date_obs in jours_val:
            subset = "val"
        else:
            subset = "test"
            
        rel_path = row["image_path"].replace("\\", "/").split("Oiseaux_Laurens/")[-1]
        src_file = LAURENS_IMG_ROOT / rel_path
        
        if not src_file.exists():
            continue
            
        label = row["label_norm"]
        dest_folder = DEST_LAURENS_SPLIT / subset / label
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(src_file, dest_folder / src_file.name)
        comptes[subset] += 1
        
    print(f"Laurens split terminé : Train={comptes['train']} | Val={comptes['val']} | Test={comptes['test']}")


def preparer_combined_split():
    print("\n--- 2. Préparation du jeu COMBINÉ (Hackaton + Laurens) ---")
    if DEST_COMBINED_SPLIT.exists():
        shutil.rmtree(DEST_COMBINED_SPLIT)
        
    for subset in ["train", "val", "test"]:
        dest_subset = DEST_COMBINED_SPLIT / subset
        dest_subset.mkdir(parents=True, exist_ok=True)
        
        # Copie depuis Hackaton
        src_hack = HACKATON_SPLIT_DIR / subset
        if src_hack.exists():
            for cls_folder in src_hack.iterdir():
                if cls_folder.is_dir():
                    (dest_subset / cls_folder.name).mkdir(parents=True, exist_ok=True)
                    for img in cls_folder.iterdir():
                        if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            shutil.copy2(img, dest_subset / cls_folder.name / f"hack_{img.name}")
                            
        # Copie depuis Laurens
        src_lau = DEST_LAURENS_SPLIT / subset
        if src_lau.exists():
            for cls_folder in src_lau.iterdir():
                if cls_folder.is_dir():
                    (dest_subset / cls_folder.name).mkdir(parents=True, exist_ok=True)
                    for img in cls_folder.iterdir():
                        if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            shutil.copy2(img, dest_subset / cls_folder.name / f"lau_{img.name}")
                            
    print("Jeu combiné créé avec succès dans :", DEST_COMBINED_SPLIT)


if __name__ == "__main__":
    preparer_laurens_split()
    preparer_combined_split()
