import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    """
    Modèle ResNet18 pour le Transfer Learning.
    On charge ResNet18 pré-entraîné et on remplace la dernière couche (fc).
    """
    modele = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = modele.fc.in_features
    modele.fc = nn.Linear(num_features, num_classes)
    return modele
