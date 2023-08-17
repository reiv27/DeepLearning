# Constants and device mode

import torch

PATH = 'data/flowers'  # Your data folder path
K = 5                  # Number of clusters or classes
DISTANCE = 'euclidean'
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
