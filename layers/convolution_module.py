import torch
import torch.nn as nn

from layers.point_convolution import PointConvolution


class ConvolutionModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(input_channels, output_channels, 5, padding=2)

    def forward(self, x):
        x0 = nn.functional.relu(self.conv3x3(x))
        x1 = nn.functional.relu(self.conv5x5(x))
        x = torch.cat([x0, x1], 1)
        return x


class FirstLayer(nn.Module):
    def __init__(self, output_channels=20):
        super().__init__()
        self.conv3x3 = PointConvolution(3, output_channels)
        self.conv5x5 = PointConvolution(5, output_channels)

    def forward(self, x):
        x0 = nn.functional.relu(self.conv3x3(x))
        x1 = nn.functional.relu(self.conv5x5(x))
        x = torch.cat([x0, x1], 1)
        return x
