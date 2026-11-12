import json
import random
import shutil
from pathlib import Path

# Fonction pour découper le dataset en 3 parties : train, val et test
def decoupage_donnees():
    # Dossier où se trouvent les images d'origine
    dossier_source = Path("/Users/grappoon/Desktop/IUT/L2/S3/R3.14 - IA/projex/Oiseaux_data/Data_Hackaton")
    # Dossier où on va copier les images triées
    dossier_destination = Path("./dataset_split")
    
    dossier_train = dossier_destination / "train"
    dossier_val = dossier_destination / "val"
    dossier_test = dossier_destination / "test"
    
    # On charge la liste des classes depuis le fichier json
    with open("classes.json", "r", encoding="utf-8") as f:
        donnees_classes = json.load(f)["classes"]
        
    # On nettoie les noms d'espèces (on remplace les espaces par des tirets du bas)
    liste_classes = []
    for c in donnees_classes:
        if c not in ("autre_oiseau", "rien", "incertain"):
            liste_classes.append(c.strip().replace(" ", "_"))
    liste_classes = sorted(list(set(liste_classes)))
    
    print(f"Nombre de classes à traiter : {len(liste_classes)}")
    
    # Si le dossier existe déjà, on le supprime pour repartir à zéro proprement
    if dossier_destination.exists():
        print("Suppression de l'ancien dossier dataset_split...")
        shutil.rmtree(dossier_destination)
        
    # On crée les dossiers de base
    dossier_train.mkdir(parents=True, exist_ok=True)
    dossier_val.mkdir(parents=True, exist_ok=True)
    dossier_test.mkdir(parents=True, exist_ok=True)
    
    # On fixe la graine pour que le mélange soit toujours le même (reproductible)
    random.seed(42)
    extensions_valides = {".jpg", ".jpeg", ".png"}
    
    nb_total_train = 0
    nb_total_val = 0
    nb_total_test = 0
    
    # On parcourt chaque espèce d'oiseau
    for espece in liste_classes:
        chemin_espece = dossier_source / espece
        if not chemin_espece.is_dir():
            print(f"Attention : dossier introuvable pour {espece}")
            continue
            
        # On récupère toutes les images de l'espèce
        images = [img for img in sorted(chemin_espece.iterdir()) if img.suffix.lower() in extensions_valides]
        random.shuffle(images)
        
        # Répartition : 70% pour train, 15% pour val, 15% pour test
        total_images = len(images)
        taille_train = int(total_images * 0.70)
        taille_val = int(total_images * 0.15)
        
        images_train = images[:taille_train]
        images_val = images[taille_train:taille_train + taille_val]
        images_test = images[taille_train + taille_val:]
        
        # On crée les sous-dossiers pour l'espèce courante
        (dossier_train / espece).mkdir(exist_ok=True)
        (dossier_val / espece).mkdir(exist_ok=True)
        (dossier_test / espece).mkdir(exist_ok=True)
        
        # On copie les fichiers vers leurs dossiers respectifs
        for img in images_train:
            shutil.copy2(img, dossier_train / espece / img.name)
            nb_total_train += 1
            
        for img in images_val:
            shutil.copy2(img, dossier_val / espece / img.name)
            nb_total_val += 1
            
        for img in images_test:
            shutil.copy2(img, dossier_test / espece / img.name)
            nb_total_test += 1
            
        print(f"  {espece:25} -> Train: {len(images_train):2d} | Val: {len(images_val):2d} | Test: {len(images_test):2d}")
        
    print("\n" + "=" * 45)
    print("Découpage terminé !")
    print(f"Train (70%) : {nb_total_train} images")
    print(f"Val   (15%) : {nb_total_val} images")
    print(f"Test  (15%) : {nb_total_test} images")
    print("=" * 45)

if __name__ == "__main__":
    decoupage_donnees()
