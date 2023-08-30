import os
from torch import is_tensor, float32
from torchvision import datasets
from config import PATH, BATCH_SIZE
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, PILToTensor, ConvertImageDtype


class FlowersDataset(Dataset):
    """Flowers Dataset."""

    def __init__(self, root_dir, transform=False):
        self.root_dir = root_dir
        self.data_list = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        image = self.data_list[idx]
        sample = read_image(f'{self.root_dir}/{image}')
        if self.transform:
            self.transform = Compose([ConvertImageDtype(float32),
                                      Resize((100, 100)),
                                      ])
        sample = self.transform(sample)

        return sample


data = FlowersDataset(PATH, transform=True)
train_dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
