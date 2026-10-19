from torch.utils.data import Dataset, DataLoader, random_split
from gensim.models import KeyedVectors
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import re
import os
import matplotlib
import matplotlib.pyplot as plt

device = "cpu"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.backends.xpu.is_available():
    device = "xpu"
else :
    "cpu"
    
print(f"Appareil utilisé : {device}")