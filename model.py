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
    def __init__(self, input_dim=300, hidden_dim=128): #dimensions des vecteurs
        super().__init__() 
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), #fonction d'activation
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() #fonction d'activation
        )
