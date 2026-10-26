from pathlib import Path
import shutil

from build_sequence import build_sequence
import pandas as pd
from PIL import Image


def get_exif_datetime(path):
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None

            # 36867 est dans le sous-répertoire Exif (0x8769).
            exif_ifd = exif.get_ifd(0x8769)
            date = (
                exif_ifd.get(36867)  # DateTimeOriginal
                or exif_ifd.get(36868)  # DateTimeDigitized
                or exif.get(306)  # DateTime dans l'IFD principal
            )

            if not date:
                return None

            return pd.to_datetime(
                date,
                format="%Y:%m:%d %H:%M:%S",
                errors="coerce",
            )
    except Exception as error:
        print(path, "erreur :", error)
        return None


def copy_images_by_sequence(df, output_folder):
    """
    Copie chaque image dans le dossier correspondant à son numéro de séquence
    """

    output_folder = Path(output_folder)

    for _, row in df.iterrows(): #parcourt les lignes du df une par une
        image = Path(row["image_path"])
        sequence_folder = output_folder / f"sequence_{row['sequence_id']}"
        #construit le chemin du dossier correspondant à la séquence
        sequence_folder.mkdir(parents=True, exist_ok=True)
        #Créer le dossier accueillant l'image + créer le dossier parent
        shutil.copy2(image, sequence_folder / image.name)
        #copie l'image dans le dossier


root = Path("../Oiseaux/Oiseaux_data/Oiseaux_Laurens")

rows = []
for img_path in root.rglob("*"):
    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        dt = get_exif_datetime(img_path)
        if dt is not None:
            rows.append({
                "image_path": str(img_path),
                "exif_datetime": dt
            })

df = pd.DataFrame(rows)
df = build_sequence(df)
df.to_csv("dataset_sequence.csv", index=False)
copy_images_by_sequence(df, "sequences")

if Path("dataset_sequence.csv").exists():
    print("Dataframe exporté avec succès")

print(df.head())
print("Nombre d'images :", len(df))
print("Nombre de séquences :", df["sequence_id"].nunique())