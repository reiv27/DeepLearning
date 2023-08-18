from datetime import datetime
from train import train
from model import model
from dataloader import train_dataloader


def main():
    start = datetime.now()
    train(train_dataloader, model)
    end = datetime.now() - start
    print(end)


if __name__ == "__main__":
    main()
