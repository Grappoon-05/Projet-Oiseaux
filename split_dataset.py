import json
import random
import shutil
from pathlib import Path

def split_dataset():
    # 1. Chemins
    src_root = Path("/Users/grappoon/Desktop/IUT/L2/S3/R3.14 - IA/projex/Oiseaux_data/Data_Hackaton")
    dest_root = Path("./dataset_split")
    
    train_dir = dest_root / "train"
    val_dir = dest_root / "val"
    test_dir = dest_root / "test"
    
    # 2. Classes cibles depuis classes.json
    with open("classes.json", "r", encoding="utf-8") as f:
        raw_classes = json.load(f)["classes"]
        
    target_classes = []
    for c in raw_classes:
        if c not in ("autre_oiseau", "rien", "incertain"):
            target_classes.append(c.strip().replace(" ", "_"))
    target_classes = sorted(list(set(target_classes)))
    
    print(f"Extraction pour {len(target_classes)} classes cibles (Train 70% / Val 15% / Test 15%).")
    
    # Réinitialisation propre du dossier de destination
    if dest_root.exists():
        print(f"Nettoyage de l'ancien {dest_root}...")
        shutil.rmtree(dest_root)
        
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(42)
    exts = {".jpg", ".jpeg", ".png"}
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for cls in target_classes:
        cls_src = src_root / cls
        if not cls_src.is_dir():
            print(f"[Attention] Dossier introuvable : {cls}")
            continue
            
        images = [f for f in sorted(cls_src.iterdir()) if f.suffix.lower() in exts]
        random.shuffle(images)
        
        # Répartition : 70% Train, 15% Val, 15% Test
        n = len(images)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        # Création des sous-dossiers par classe
        (train_dir / cls).mkdir(exist_ok=True)
        (val_dir / cls).mkdir(exist_ok=True)
        (test_dir / cls).mkdir(exist_ok=True)
        
        for img in train_imgs:
            shutil.copy2(img, train_dir / cls / img.name)
            total_train += 1
            
        for img in val_imgs:
            shutil.copy2(img, val_dir / cls / img.name)
            total_val += 1
            
        for img in test_imgs:
            shutil.copy2(img, test_dir / cls / img.name)
            total_test += 1
            
        print(f"  {cls:25} -> Train: {len(train_imgs):2d} | Val: {len(val_imgs):2d} | Test: {len(test_imgs):2d}")
        
    print("\n" + "=" * 50)
    print(f"Séparation 3-splits terminée avec succès !")
    print(f"Total Train (70%) : {total_train} images")
    print(f"Total Val   (15%) : {total_val} images")
    print(f"Total Test  (15%) : {total_test} images")
    print("=" * 50)

if __name__ == "__main__":
    split_dataset()
