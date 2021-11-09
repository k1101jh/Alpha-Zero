import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroModel(nn.Module):
    def __init__(self, in_channels, mid_channels=128, num_blocks=20, board_size=15):
        super().__init__()
        self.conv = ConvBnReluBlock(in_channels, mid_channels)
        self.layers = self._make_layer(mid_channels, num_blocks)
        self.policy = PolicyHead(mid_channels, board_size)
        self.value = ValueHead(mid_channels, board_size)

    def _make_layer(self, in_channels, num_blocks):
        layers = []
        for num_block in range(num_blocks):
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
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels))
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.residual_block(x)
        out = out + x
        out = self.relu(out)
        return out


class PolicyHead(nn.Module):
    def __init__(self, in_channels, board_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(board_size**2 * 4, board_size**2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.log_softmax(out, dim=1)
        return out


class ValueHead(nn.Module):
    def __init__(self, in_channels, board_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(board_size**2, 64)
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
        print(out.shape())
        return out
