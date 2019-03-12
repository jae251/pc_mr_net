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
        self._padding_size = (kernel_size - 1) * .5
        self.pad = nn.ZeroPad2d(self._padding_size)
        self.kernel_coordinates = self.compute_kernel_coordintes()
        self.conv_1x1 = nn.Conv2d(24, output_features, kernel_size=1)

    def forward(self, x):
        shape_0, shape_1 = x.shape
        x_padded = self.pad(x)
        channels = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if i == self._middle_coordinate and j == self._middle_coordinate:
                    continue
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
        coordinates = coordinates[np.all(coordinates != 0, axis=1)]
        return coordinates
