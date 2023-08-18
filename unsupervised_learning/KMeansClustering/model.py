# Model of KMeaningClustering

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

        self.filters = 1024

        self.fc = nn.Sequential(
            nn.Linear(1024, 169),
            nn.PReLU(),
            nn.Linear(169, 2)
        )

    def forward(self, x):
        #print(x.shape)
        x = self.conv(x)
        #print(x.shape)
        x = x.view(-1, self.filters)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x


'''
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
'''

model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), LR)
