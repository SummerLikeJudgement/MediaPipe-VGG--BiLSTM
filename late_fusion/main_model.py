import torch
from torch import nn
from torch.nn import functional as F
from deep_feature.VGG16 import VGG
from geometry_feature.BiLSTM import LSTM

class denselayer(nn.Module):
    def __init__(self, channels):
        super(denselayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(channels,512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 5),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(5),
            nn.Dropout(0.4)
        )
    def forward(self, x):
        x = self.layer(x)
        return x


class LateFusion(nn.Module):
    def __init__(self):
        super(LateFusion, self).__init__()
        self.dense = denselayer(128+512)
    # dep(batch_size, 512, 1, 1) geo(batch_size, 128, 1)
    def forward(self, dep_feature, geo_feature):
        dep_feature = dep_feature.squeeze(-1).squeeze(-1)# (batch_size, 512)
        geo_feature = geo_feature.squeeze(-1)# (batch_size, 128)
        fusion_feature = torch.cat((dep_feature, geo_feature), 1)# (batch_size, 128+512)
        out = self.dense(fusion_feature)
        out = F.softmax(out, dim=1)
        return out

if __name__ == '__main__':
    model = LateFusion()
    dep = torch.randn(2, 512, 1, 1)
    geo = torch.randn(2, 128, 1)
    out = model(dep, geo)
    print(out.shape)


