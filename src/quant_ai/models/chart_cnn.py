"""CNN compacta para ventanas OHLC de 48 o 96 velas."""
from torch import nn

class ChartCNN(nn.Module):
    def __init__(self, window=48):
        super().__init__()
        if window not in (48,96):
            raise ValueError("La ventana debe ser 48 o 96.")
        layers = []
        previous = 1
        for channels in (32,64,128):
            layers.extend([nn.Conv2d(previous,channels,(5,3),padding=(2,1)),
                           nn.BatchNorm2d(channels),nn.LeakyReLU(.01),nn.MaxPool2d((2,1))])
            previous = channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((2,6)),nn.Flatten(),nn.Dropout(.5),nn.Linear(1536,1))
    def forward(self,x):
        return self.head(self.features(x)).squeeze(-1)
