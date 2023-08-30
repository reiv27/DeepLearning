# Constants and device mode

import torch

PATH = 'data/flowers'  # Your data folder path
K = 5                  # Number of clusters or classes
BATCH_SIZE = 64
EPOCHS = 1
LR = 0.001
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
