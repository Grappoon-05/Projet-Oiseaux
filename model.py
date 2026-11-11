# Source officielle PyTorch Transfer Learning Tutorial :
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Discussion PyTorch forum sur le remplacement de model.fc :
# https://discuss.pytorch.org/t/how-to-modify-the-final-fc-layer-based-on-the-torch-model/766

import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    """
    Modèle ResNet18 pour le Transfer Learning.
    Pattern classique documenté sur PyTorch.org :
    - On charge ResNet18 pré-entraîné
    - On récupère num_features via model.fc.in_features
    - On remplace model.fc par un nouveau nn.Linear(num_features, num_classes)
    """
    # 1. Poids pré-entraînés ImageNet par défaut
    modele = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 2. Remplacement de la couche de classification (fc)
    num_features = modele.fc.in_features
    modele.fc = nn.Linear(num_features, num_classes)

    return modele
