# 참고: https://github.com/reinforcement-learning-kr/alpha_omok/blob/master/2_AlphaOmok/model.py

import torch.nn as nn


class AlphaZeroModel(nn.Module):
    def __init__(self, in_channels, mid_channels=128, num_blocks=7, board_size=15):
        """[summary]
            Alpha Zero network.
        Args:
            in_channels (int): Num of input channels.
            mid_channels (int, optional): Num of mid channels. Defaults to 128.
            num_blocks (int, optional): Num of residual blocks. Defaults to 7.
            board_size (int, optional): Size of board. Defaults to 15.
        """
        super().__init__()
        self.conv = ConvBnReluBlock(in_channels, mid_channels)
        self.layers = self._make_layer(mid_channels, num_blocks)
        self.policy = PolicyHead(mid_channels, board_size)
        self.value = ValueHead(mid_channels, board_size)

    @classmethod
    def _make_layer(cls, in_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.layers(out)
        policy = self.policy(out)
        value = self.value(out)
        return [policy, value]


class ConvBnReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """[summary]
            Convolution block with batchnorm and ReLU.
        Args:
            in_channels (int): Num of input channels.
            out_channels (int): Num of output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        """[summary]
        Args:
            in_channels (int): Num of input channels.
        """
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.residual_block(x)
        out = out + x
        out = self.relu(out)
        return out


class PolicyHead(nn.Module):
    def __init__(self, in_channels, board_size):
        """[summary]
            Policy network.
            Returns priors of branches.
        Args:
            in_channels (int): Num of input channels.
            board_size (int): Size of board.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(board_size**2 * 4, board_size**2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.softmax(out)
        return out


class ValueHead(nn.Module):
    def __init__(self, in_channels, board_size):
        """[summary]
            Value network.
            Returns value of state.
        Args:
            in_channels (int): Num of input channels.
            board_size (int): Size of board.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(board_size * board_size * 2, 64)
        self.linear2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.tanh(out)
        return out
