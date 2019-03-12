import torch
import torch.nn as nn


# m = torch.randn(1, 3, 100, 50)
# print(m)
#
# m_padded = torch.nn.ZeroPad2d(1)(m)
# print(m_padded.shape)
# print(m.shape)
# m0 = m_padded[:, :, :-2, 0:-2] - m
# m1 = m_padded[:, :, :-2, 1:-1] - m
# m2 = m_padded[:, :, :-2, 2:] - m
# m3 = m_padded[:, :, 1:-1, 0:-2] - m
# m4 = m_padded[:, :, 1:-1, 2:] - m
# m5 = m_padded[:, :, 2:, 0:-2] - m
# m6 = m_padded[:, :, 2:, 1:-1] - m
# m7 = m_padded[:, :, 2:, 2:] - m
#
# print(torch.cat([m0, m1, m2, m3, m4, m5, m6, m7], 1).shape)


class PointConvolution(nn.Module):
    '''
    Graph convolution over points with relative position as features
    and adjacency information from sensor picture format and 3x3 kernel.
    In other words:
    2d Convolution with kernel size 3x3, where the middle value is subtracted
    from the outer values and is otherwise ignored.
    '''

    def __init__(self, output_features):
        super().__init__()
        self.output_features = output_features
        self.conv_1x1 = nn.Conv2d(24, output_features, kernel_size=1)

    def forward(self, x):
        x_padded = nn.ZeroPad2d(1)(x)
        x0 = x_padded[:, :, :-2, :-2] - x
        x1 = x_padded[:, :, :-2, 1:-1] - x
        x2 = x_padded[:, :, :-2, 2:] - x
        x3 = x_padded[:, :, 1:-1, :-2] - x
        x4 = x_padded[:, :, 1:-1, 2:] - x
        x5 = x_padded[:, :, 2:, :-2] - x
        x6 = x_padded[:, :, 2:, 1:-1] - x
        x7 = x_padded[:, :, 2:, 2:] - x
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], 1)
        x = self.conv_1x1(x)
        return x
