import torch.nn as nn
from torch.nn.functional import relu

from layers.convolution_module import FirstLayer, ConvolutionModule


class PointCloudMapRegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FirstLayer(10)
        self.conv2 = ConvolutionModule(20, 10)
        self.conv3 = ConvolutionModule(20, 10)
        self.conv4 = ConvolutionModule(20, 10)
        self.conv5 = ConvolutionModule(20, 10)
        self.conv6 = ConvolutionModule(20, 10)
        self.conv7 = ConvolutionModule(20, 10)
        self.conv8 = ConvolutionModule(20, 10)
        self.conv9 = ConvolutionModule(20, 10)
        self.conv10 = ConvolutionModule(20, 10)
        self.conv11 = ConvolutionModule(20, 10)
        self.conv12 = ConvolutionModule(20, 10)
        self.regression = nn.Conv2d(20, 3, 1)

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
        x = relu(self.conv10(x))
        x = relu(self.conv11(x))
        x = relu(self.conv12(x))
        x = self.regression(x)
        return x
