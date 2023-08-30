# Model oсf KMeaningClustering

from torch import nn, optim
from config import K, DEVICE, LR


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, 3),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, 3),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )

        self.filters = 3200

        self.fc = nn.Sequential(
            nn.Linear(self.filters, 169),
            nn.PReLU(),
            nn.Linear(169, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        #print(f'after conv: {x.shape}')
        x = x.view(-1, self.filters)
        #print(f'after x.flatten: {x.shape}')
        x = self.fc(x)
        #print(f'after fc: {x.shape}')
        return x


model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), LR)
