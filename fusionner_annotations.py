import csv
from pathlib import Path


IMAGES_FILE = Path("dataset_sequence.csv")
ANNOTATIONS_FILE = Path("annotations_sequences.csv")
OUTPUT_FILE = Path("dataset_annote.csv")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        sample = file.read(4096)
        file.seek(0)
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;").delimiter
        return list(csv.DictReader(file, delimiter=delimiter))


def main():
    if not IMAGES_FILE.exists():
        raise FileNotFoundError(f"Fichier absent : {IMAGES_FILE}")
    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier absent : {ANNOTATIONS_FILE}. "
            "Exporte l'Excel en CSV UTF-8 sous ce nom."
        )

    images = read_csv(IMAGES_FILE)
    annotations = read_csv(ANNOTATIONS_FILE)
    if not images or not annotations:
        raise ValueError("Un des CSV est vide.")

    required = {"sequence_id", "label"}
    if not required.issubset(annotations[0]):
        raise ValueError("Le CSV d'annotations doit contenir sequence_id et label.")

    by_sequence = {}
    for row in annotations:
        sequence_id = int(row["sequence_id"])
        label = row["label"].strip()
        if not label:
            raise ValueError(f"Label vide pour la séquence {sequence_id}.")
        if sequence_id in by_sequence:
            raise ValueError(f"Annotation dupliquée pour la séquence {sequence_id}.")
        by_sequence[sequence_id] = label

    annotated_images = []
    for row in images:
        sequence_id = int(row["sequence_id"])
        if sequence_id in by_sequence:
            row["label"] = by_sequence[sequence_id]
            annotated_images.append(row)

    image_sequences = {int(row["sequence_id"]) for row in images}
    held_out = sorted(image_sequences - by_sequence.keys())
    print(
        f"{len(annotated_images)} image(s) annotée(s) pour l'entraînement. "
        f"{len(held_out)} séquence(s) conservée(s) sans annotation pour le test."
    )

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(annotated_images[0]))
        writer.writeheader()
        writer.writerows(annotated_images)

    print(f"{OUTPUT_FILE} créé avec {len(annotated_images)} images annotées.")


if __name__ == "__main__":
    main()
