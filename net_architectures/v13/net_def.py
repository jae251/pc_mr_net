import torch.nn as nn
from torch.nn.functional import relu

from layers.convolution_module import FirstLayerZeroed, ConvolutionModule


class PointCloudMapRegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FirstLayerZeroed(20)
        self.conv2 = ConvolutionModule(40, 20)
        self.conv3 = ConvolutionModule(40, 20)
        self.conv4 = ConvolutionModule(40, 20)
        self.conv5 = ConvolutionModule(40, 20)
        self.conv6 = ConvolutionModule(40, 20)
        self.regression = nn.Conv2d(40, 3, 1)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = relu(self.conv4(x))
        x = relu(self.conv5(x))
        x = relu(self.conv6(x))
        x = self.regression(x)
        return x
