import torch
import torch.nn as nn
from torch.nn.functional import relu

from layers.convolution_module import FirstLayer, ConvolutionModule


class ConvolutionModule_mod(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(input_channels, output_channels, 5, padding=2)
        self.conv11x11 = nn.Conv2d(input_channels, output_channels, 11, padding=5)

    def forward(self, x):
        x0 = nn.functional.relu(self.conv3x3(x))
        x1 = nn.functional.relu(self.conv5x5(x))
        x2 = nn.functional.relu(self.conv11x11(x))
        x = torch.cat([x0, x1, x2], 1)
        return x


class PointCloudMapRegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FirstLayer()
        self.conv2 = ConvolutionModule_mod(40, 64)
        self.conv2 = ConvolutionModule(192, 40)
        self.conv3 = ConvolutionModule(80, 20)
        self.conv4 = ConvolutionModule(40, 20)
        self.conv5 = ConvolutionModule(40, 20)
        self.conv6 = ConvolutionModule(40, 20)
        self.conv7 = ConvolutionModule(40, 20)
        self.conv8 = ConvolutionModule(40, 20)
        self.conv9 = ConvolutionModule(40, 20)
        self.regression = nn.Conv2d(40, 3, 1)

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
