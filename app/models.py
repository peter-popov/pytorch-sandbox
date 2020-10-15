import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualLayer(nn.Module):
    def __init__(self, inchannels, outchannels, stride = 1):
        super(ResidualLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(inchannels),
            nn.ReLU(),
            nn.Conv2d(inchannels, outchannels, 3, padding = 1, stride = stride),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(),
            nn.Conv2d(outchannels, outchannels, 3, padding = 1),
        )
        if (inchannels != outchannels or stride > 1):
            self.projection = nn.Conv2d(inchannels, outchannels, kernel_size = 1, stride = stride)
        else:
            self.projection = None

    def forward(self, x):
        if (self.projection):
            return self.layer(x) + self.projection(x)
        else:
            return self.layer(x) + x

class SampleModel(nn.Module):
    def __init__(self, num_classes):
        super(SampleModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding = 1), # 32x32
            ResidualLayer(16, 96, stride = 2), 
            ResidualLayer(96, 96),
            ResidualLayer(96, 96),
            ResidualLayer(96, 192, stride = 2),
            ResidualLayer(192, 192),
            ResidualLayer(192, 192),
            ResidualLayer(192, 384, stride = 2),
            ResidualLayer(384, 384),
            ResidualLayer(384, 384),  
            #nn.AdaptiveAvgPool2d(1)
        )

        # for m in self.cnn:
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        self.fc = nn.Sequential(
            nn.BatchNorm2d(384), 
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(), 
            nn.Linear(384, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x) 
        return x


class SampleVggStyle(nn.Module):
    def __init__(self, num_classes):
        super(SampleVggStyle, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding =1), nn.ReLU(), #32x32
            nn.Conv2d(32, 32, 3, padding =1), nn.ReLU(),
            nn.MaxPool2d(2),                            #16x16
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, 3, padding =1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding =1), nn.ReLU(),
            nn.MaxPool2d(2),                            #8x8
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, 3, padding =1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding =1), nn.ReLU(),
            nn.MaxPool2d(2),                            #4x4
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(128 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.cnn(x)
        return x