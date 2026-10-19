import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
import torch


IMAGE_EXTENSIONS = {".jpg", ".png"}




# On récupère le chemins des images
def list_images(root):
    """Return (image paths, labels, class names) from a folder-per-class tree."""
    root = Path(root)

    # Les classes sont les noms des sous-dossiers. On les trie pour que
    # "Parus_major" ait toujours le même numéro d'un lancement à l'autre.
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())

    # Deux listes de même longueur : paths[i] est une image, labels[i] est le numéro de sa classe
    paths, labels = [], []

    # enumerate donne le numéro de la classe (0, 1, 2...) en plus de son nom.
    # Le réseau ne manipule que des nombres : c'est ce numéro qui sert de label.
    for label, class_name in enumerate(classes):
        for file in sorted((root / class_name).iterdir()):
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(file)
                labels.append(label)
            else:
                print(f"Error, image extension is {file.suffix}")
                # essayez de regarder les images qui posent problème
    return paths, labels, classes


# Charger les images
def load_image(path):
    return Image.open(path).convert("RGB")


# fonction pas obligétoire, juste pour vérifier que votre chargement marche bien
def show_random_image(dataset):
    idx = random.randint(0, len(dataset) - 1)
    image, label = dataset[idx]
    print(f"Image {idx}/{len(dataset)} : {paths[idx]}")
    plt.imshow(image.permute(1, 2, 0))
    plt.title(classes[label])
    plt.axis("off")
    plt.show()


# Transformation en tenseur, PENSEZ BIEN à redimentionner vos images pour qu'elles aient toutes la même taille !
default_transform = transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToTensor()
])


class OiseauxDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform=default_transform):
        # On garde seulement les chemins : 13'000 images chargées d'un coup, ce serait beaucoup trop lourd en mémoire.
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # L'image n'est lue sur le disque qu'au moment où on la demande
        image = load_image(self.paths[idx])
        return self.transform(image), self.labels[idx]




if __name__ == "__main__":
    data_root = "Oiseaux_data/Data_Hackaton/"

    # on récupère les chemins des images, les labels et les classes
    paths, labels, classes = list_images(data_root)
    print(f"{len(paths)} images, {len(classes)} classes")

    # on charge le dataset et on le met dans un DataLoader
    dataset = OiseauxDataset(paths, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # on récupère un batch d'images et de labels pour vérifier que tout fonctionne bien.
    images, targets = next(iter(loader))
    print(images.shape, [classes[t] for t in targets[:5]])

    # on affiche une image random
    show_random_image(dataset)