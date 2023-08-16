import os
import random
import numpy as np
from config import *
from torch import tensor
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


training_data = datasets.ImageFolder(
    root=TRAIN,
    transform=Compose([
        Resize((180, 200)),
        ToTensor(),
        #Normalize(mean, std)
    ])
)


train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
