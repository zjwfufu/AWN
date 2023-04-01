"""
part of codes is adopted from:
https://github.com/mxbastidasr/DAWN_WACV2020
"""

import torch.nn as nn


class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

        self.conv_even = lambda x: x[:, :, ::2]
        self.conv_odd = lambda x: x[:, :, 1::2]

    def forward(self, x):
        """
        returns the odd and even part
        :param x:
        :return: x_even, x_odd
        """
        return self.conv_even(x), self.conv_odd(x)


class Operator(nn.Module):
    def __init__(self, in_planes, kernel_size=3, dropout=0.):
        super(Operator, self).__init__()

        pad = (kernel_size - 1) // 2 + 1

        self.operator = nn.Sequential(
            nn.ReflectionPad1d(pad),
            nn.Conv1d(in_planes, in_planes,
                      kernel_size=(kernel_size,), stride=(1,)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(in_planes, in_planes,
                      kernel_size=(kernel_size,), stride=(1,)),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Operator as Predictor() or Updator()
        :param x:
        :return: P(x) or U(x)
        """
        x = self.operator(x)
        return x


class LiftingScheme(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(LiftingScheme, self).__init__()

        self.split = Splitting()

        self.P = Operator(in_planes, kernel_size)
        self.U = Operator(in_planes, kernel_size)

    def forward(self, x):
        """
        Implement Lifting Scheme
        :param x:
        :return: c: approximation coefficient
                 d: details coefficient
        """
        (x_even, x_odd) = self.split(x)
        c = x_even + self.U(x_odd)
        d = x_odd - self.P(c)
        return c, d
