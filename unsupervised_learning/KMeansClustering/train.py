import os
import torch
import numpy as np
from model import *
from tkinter import *
from torch import no_grad, tensor
from datetime import datetime
import matplotlib.pyplot as plt
from config import *
from dataloader import train_dataloader


def train(dataloader, model):
    for batch, data in enumerate(train_dataloader):
        data = data.to(DEVICE)
        x = model(data)
        x = x.cpu().detach().numpy()


train(train_dataloader, model)
