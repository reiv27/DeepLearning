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


def train(dataloader, model, loss_fn, optimizer):
    now = datetime.now()
    os.mkdir(f'plots/{now}')
    config_file = open(f'plots/{now}/CONFIG_{now}.log', 'w')
    config_file.write(f'device = {DEVICE}\nbatch_size = {BATCH_SIZE}\nepochs = {EPOCHS}\nlr = {LR}\ntriplets = {TYPE}\nminer_margin = {MINER_MARGIN}\nloss_miner = {LOSS_MARGIN}\n\n')
    config_file.close()
    size = len(dataloader.dataset)
    model.train()
    sum = 0
    for t in range(EPOCHS):
        running_loss = []
        model.train()
        for batch, (data, labels) in enumerate(dataloader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            embeddings = model(data)
            #print(f'data = {embeddings.shape}, labels = {labels.shape}')
            triplet = miner_fn(embeddings, labels)
            loss = loss_fn(embeddings, labels, triplet)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
        
        print(f"Epoch: {t+1}/{EPOCHS} - Loss: {np.mean(running_loss)}")
        config_file = open(f'plots/{now}/CONFIG_{now}.log', 'a')
        config_file.write(f'Epoch: {t+1}/{EPOCHS} - Loss: {np.mean(running_loss)}\n')
        config_file.close()
        if t % 10 == 0:
            train_results = []
            labels = []
            model.eval()
            with no_grad():
                for img, label in train_dataloader:
                    train_results.append(model(img.to(DEVICE)).cpu().numpy())
                    labels.append(label)

                train_results = np.concatenate(train_results)
                labels = np.concatenate(labels)

                plt.figure(figsize=(15, 10))
                for label in np.unique(labels):
                    tmp = train_results[labels==label]
                    plt.scatter(tmp[:, 0], tmp[:, 1], label=label)

                plt.legend()
                plt.savefig(f'plots/{now}/{now}_epoch-{t}.jpg')
                plt.close()
