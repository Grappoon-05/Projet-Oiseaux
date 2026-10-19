from torch.utils.data import Dataset, DataLoader, random_split 
from gensim.models import KeyedVectors 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import torch
import re
import os

class Classifier(nn.Module): #Implémentation du classifieur linéaire simple
    def __init__(self, input_dim=..., hidden_dim=...): #dimensions des vecteurs
        super().__init__() 
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), #fonction d'activation
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() #fonction d'activation
        )


classifieur = Classifier()

optimizer = torch.optim.Adam(Classifier.parameters(), lr=0.01)
loss_ = nn.BCELoss()

for epoch in range(10):

    loss_actuelles = 0.0
    for i, data in enumerate(train_loader, 0):
        input, labels = data #Récupère les entrées et les labes dans une liste

    optimizer.zero_grad()
    #remise à zéro des gradients

    sortie = Classifier(input)
    loss = loss_(sortie, labels)
    loss.backward()
    optimizer.step()
    #propagation et backpropagation + opti

    #affichage de la loss
    loss_actuelles += loss.item()
    if i % 2000 == 1999: #affiche tout les 2000 batch
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss_actuelles / 2000:.3f}')
        loss_actuelles = 0.0

print('Entrainement terminé')