import torch
import torch.nn as nn

from layers.point_convolution import PointConvolution


class ConvolutionModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super.__init__()
        self.conv3x3 = nn.Conv2d(input_channels, output_channels, 3)
        self.conv5x5 = nn.Conv2d(input_channels, output_channels, 5)

    def forward(self, x):
        x0 = nn.functional.relu(self.conv3x3(x))
        x1 = nn.functional.relu(self.conv5x5(x))
        x = torch.cat([x0, x1])
        return x


class FirstLayer(nn.Module):
    def __init__(self):
        super.__init__()
        self.conv3x3 = PointConvolution(3, 20)
        self.conv5x5 = PointConvolution(5, 20)

    def forward(self, x):
        x0 = nn.functional.relu(self.conv3x3(x))
        x1 = nn.functional.relu(self.conv5x5(x))
        x = torch.cat([x0, x1])
        return x
