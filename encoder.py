import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(
        self,
        in_dim=11,
        emb_dim=64
    ):
        super(Encoder, self).__init__()
        self.convs = nn.Sequential( # 96 x 64
            nn.Conv2d(in_dim, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2), # 48 x 32
            
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2), # 24 x 16

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2), # 12 x 8

            nn.Conv2d(64, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d(2), # 6 x 4
        )
        
        self.fcs = nn.Sequential(
            nn.Linear(192, emb_dim * 2),
            nn.ReLU(True),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        btz = x.shape[0]
        x = self.convs(x)
        x = x.reshape(btz, -1)
        x = self.fcs(x)
        return x
