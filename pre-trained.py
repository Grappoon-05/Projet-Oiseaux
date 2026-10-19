from torchvision import models

modele = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
modele.fc = torch.nn.Linear(modele.fc.in_features, len(classes))