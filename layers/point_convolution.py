import numpy as np
import torch
import torch.nn as nn


class PointConvolution(nn.Module):
    '''
    Graph convolution over points with relative position as features
    and adjacency information from sensor picture format and quadratic kernel.
    In other words:
    2d Convolution with quadratic kernel, where the middle value is subtracted
    from the outer values and is otherwise ignored.
    '''

    def __init__(self, kernel_size, output_features):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_features = output_features
        self._padding_size = int((kernel_size - 1) * .5)
        self.pad = nn.ZeroPad2d(self._padding_size)
        self.kernel_coordinates = self._compute_kernel_coordinates()
        self.conv_1x1 = nn.Conv2d(3 * len(self.kernel_coordinates), output_features, kernel_size=1)

    def forward(self, x):
        _, _, shape_0, shape_1 = x.shape
        x_padded = self.pad(x)
        channels = []
        for i, j in self.kernel_coordinates:
            channel = x_padded[:, :, i:shape_0 + i, j:shape_1 + j] - x
            channels.append(channel)
        x = torch.cat(channels, 1)
        x = self.conv_1x1(x)
        return x

    def _compute_kernel_coordinates(self):
        i, j = np.meshgrid(np.arange(self.kernel_size), np.arange(self.kernel_size))
        i = i.reshape(-1)
        j = j.reshape(-1)
        coordinates = np.dstack((i, j))[0]
        coordinates = coordinates[np.any(coordinates != self._padding_size, axis=1)]
        return coordinates


class PointConvolutionZeroed(nn.Module):
    '''
    Graph convolution over points with relative position as features
    and adjacency information from sensor picture format and quadratic kernel.
    In other words:
    2d Convolution with quadratic kernel, where the middle value is subtracted
    from the outer values and is otherwise ignored.
    '''

    def __init__(self, kernel_size, output_features):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_features = output_features
        self._padding_size = int((kernel_size - 1) * .5)
        self.pad = nn.ZeroPad2d(self._padding_size)
        self.kernel_coordinates = self._compute_kernel_coordinates()
        self.conv_1x1 = nn.Conv2d(3 * len(self.kernel_coordinates), output_features, kernel_size=1)

    def forward(self, x):
        _, _, shape_0, shape_1 = x.shape
        x_padded = self.pad(x)
        mask = (x_padded != 0).float()
        channels = []
        for i, j in self.kernel_coordinates:
            # calc each relative coord for each point selected by kernel
            channel = x_padded[:, :, i:shape_0 + i, j:shape_1 + j] - x
            # Features are scaled with this approach, but invalid points, represented as (0,0,0),
            # can potentially create distortions. Therefore set these features to 0
            channel *= mask[:, :, i:shape_0 + i, j:shape_1 + j]
            channels.append(channel)
        x = torch.cat(channels, 1)
        x = self.conv_1x1(x)
        return x

    def _compute_kernel_coordinates(self):
        i, j = np.meshgrid(np.arange(self.kernel_size), np.arange(self.kernel_size))
        i = i.reshape(-1)
        j = j.reshape(-1)
        coordinates = np.dstack((i, j))[0]
        coordinates = coordinates[np.any(coordinates != self._padding_size, axis=1)]
        return coordinates
