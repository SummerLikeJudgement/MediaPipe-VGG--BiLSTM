import torch
import torch.nn as nn
from torchvision import models


# SE块
class selayer(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super(selayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 全连接块
class denselayer(nn.Module):
    def __init__(self, channel):
        super(denselayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel),
            nn.ReLU(),
            nn.BatchNorm1d(channel)
        )
    def forward(self, x):
        x = self.layer(x)
        return x

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = models.vgg16(pretrained=False).features # 只使用全卷积层
        self.avgpool = nn.AdaptiveMaxPool2d((1,1)) # 全局平均池化
        self.dense = denselayer(512) # 全连接
        self.se = selayer(512) # SE
    def forward(self, x):
        x = self.vgg(x) # (batch_size, 512, 7, 7)
        x = self.avgpool(x) # (batch_size, 512, 1, 1)
        x = self.dense(x.view(x.size(0), -1)) # (batch_size, 512)且batch_size>1
        x = self.se(x.unsqueeze(-1).unsqueeze(-1)) # (batch_size, 512, 1, 1)
        return x

if __name__ == "__main__":
    vgg = VGG()
    tensor = torch.randn(2, 3, 224, 224)
    print(vgg(tensor).shape)
