#import os
import torch
import numpy as np
from model import model
#from tkinter import *
#from torch import no_grad, tensor
#from datetime import datetime
import matplotlib.pyplot as plt
from config import DEVICE
from dataloader import train_dataloader


def train(dataloader, model):
    results = []
    for batch, data in enumerate(train_dataloader):
        data = data.to(DEVICE)
        x = model(data)
        x = x.cpu().detach().numpy()
        results.append(x)

    results = np.concatenate(results)

    plt.figure(figsize=(15, 10))
    for i in range(len(results)):
        plt.scatter(results[i][0], results[i][1])

    plt.legend()
    plt.savefig('test.jpg')
    plt.close()

train(train_dataloader, model)
