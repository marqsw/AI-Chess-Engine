"""
Due to the limited understanding of neural network, I have referenced the code from https://github.com/michaelnny/alpha_zero/blob/main/alpha_zero/core/network.py
"""


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


class ResNetBlock(nn.Module):
    """
    Defines the basic residual block for use in the ChessModel
    """

    def __init__(
            self,
            num_filters: int,
    ) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        ).to(device)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
        ).to(device)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class ChessModel(nn.Module):
    def __init__(self, input_shape: tuple = (486, 8, 8)):
        """
        Initialise the chess model
        @param input_shape: the shape of tensor the network would take in.
        """
        super().__init__()

        c, h, w = input_shape

        conv_out_hw = calc_conv2d_output((h, w), 3, 1, 1)
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        ).to(device)

        # Residual blocks
        res_blocks = []
        for _ in range(19):
            res_blocks.append(ResNetBlock(256))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, 73),
        ).to(device)

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        ).to(device)

        self.to(device)

    def forward(self, x: torch.Tensor):
        """
        Do a forward pass through the neural net.
        @param x: input tensor
        @return: output of the policy network, and value network
        """
        conv_block_out = self.conv_block(x.to(device))
        features = self.res_blocks(conv_block_out)
        pi_logits = F.softmax(self.policy_head(features), dim=1)
        value = self.value_head(features)

        return pi_logits, value

    def predict(self, x):
        """
        Do a forward pass through the neural net without gradient.
        @param x: input tensor
        @return: output of the policy network, and value network
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

