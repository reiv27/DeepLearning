from datetime import datetime
from model import *
from train import train
from dataloader import train_dataloader


def main():
    start = datetime.now()
    train(train_dataloader, model, loss_fn, optimizer)
    end = datetime.now() - start
    print(end)


if __name__=="__main__":
    main()
