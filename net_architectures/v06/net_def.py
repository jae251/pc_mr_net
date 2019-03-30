import torch.nn as nn
from torch.nn.functional import relu

from layers.convolution_module import FirstLayer, ConvolutionModule


class PointCloudMapRegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FirstLayer(40)
        self.conv2 = ConvolutionModule(80, 40)
        self.conv3 = ConvolutionModule(80, 40)
        self.conv4 = ConvolutionModule(80, 40)
        self.conv5 = ConvolutionModule(80, 40)
        self.conv6 = ConvolutionModule(80, 40)
        self.conv7 = ConvolutionModule(80, 40)
        self.conv8 = ConvolutionModule(80, 40)
        self.conv9 = ConvolutionModule(80, 40)
        self.regression = nn.Conv2d(80, 3, 1)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        x = relu(self.conv5(x))
        x = relu(self.conv6(x))
        x = relu(self.conv7(x))
        x = relu(self.conv8(x))
        x = relu(self.conv9(x))
        x = self.regression(x)
        return x
