from torchvision import models
import torch

# Extrait exact du sujet (page 5) :
# from torchvision import models
# modele = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# modele.fc = torch.nn.Linear(modele.fc.in_features, len(classes))

def get_model(num_classes):
    """Charge ResNet18 pré-entraîné avec la tête adaptée aux classes du sujet."""
    modele = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    modele.fc = torch.nn.Linear(modele.fc.in_features, num_classes)
    return modele
