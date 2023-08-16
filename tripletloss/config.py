import torch

TRAIN = "data/my_data_2/train"  # Your training data path
TEST = "data/my_data/test"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
BATCH_SIZE = 128
EPOCHS = 1000
LR = 0.00005
TYPE = "semihard"  # hard, semihard, easy, all
MINER_MARGIN = 1
LOSS_MARGIN = 5
