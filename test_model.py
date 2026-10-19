from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import re
import os
import matplotlib
import matplotlib.pyplot as plt
from selection_device import device_config

device = device_config

model = torch.load("model", weights_only=False)
model = model.to(device)
model.eval()
print("Modèle chargé")

idx = random.randint(0, len(test_data) - 1)
X, true_label = test_data[idx]  # accès direct par index, pas besoin du DataLoader ici
inputs = X.unsqueeze(0).to(device)  # ajoute la dim batch : (3,32,32) -> (1,3,32,32)

with torch.no_grad():
    pred = model(inputs)

predicted_class = pred.argmax(dim=1).item()
print(f"Prédit : {labels_key[predicted_class]})