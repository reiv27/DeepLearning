import os
from torchvision import datasets
from torch import tensor, is_tensor
from config import PATH, BATCH_SIZE
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class FlowersDataset(Dataset):
    """Flowers Dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.data_list = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        image_name = self.data_list[idx]
        sample = read_image(f'{self.root_dir}/{image_name}')

        if self.transform:
            sample = self.transform(sample)

        return sample


'''
transform = Compose([
    Resize((180, 200)),
    ToTensor(),
    #Normalize(mean, std)
])
'''

data = FlowersDataset(PATH, Resize((224, 224)))

print(data[0])


train_dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

