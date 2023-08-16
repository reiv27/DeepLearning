from torch import nn, optim, jit, Tensor, relu
from config import DEVICE, LR, TYPE, MINER_MARGIN, LOSS_MARGIN
from pytorch_metric_learning import miners, losses, distances, reducers, testers
from torchsummary import summary

class ConvNet(nn.Module):
    def __init__(self, emb_dim=128, k1sz=3, k2sz=3, k3sz=3, filters=1024):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, k1sz),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, k2sz),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 32, k3sz),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self._filters = filters
        self.fc = nn.Sequential(
            nn.Linear(self._filters, 128),
            nn.PReLU(),
            nn.Linear(128, emb_dim)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self._filters)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

            
model = ConvNet(2, 3, 5, 3, 32*20*22).to(DEVICE)
model.apply(init_weights)

distance = distances.LpDistance(normalize_embeddings=True, power=2) 
reducer = reducers.MeanReducer()
miner_fn = miners.TripletMarginMiner(margin=MINER_MARGIN, distance=distance, type_of_triplets=TYPE).to(DEVICE)
loss_fn = losses.TripletMarginLoss(margin=LOSS_MARGIN, distance=distance, reducer=reducer).to(DEVICE)
optimizer = optim.Adam(model.parameters(), LR)
