"""LeNet-5 style model used by the EMNIST experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TrainableAveragePool2d(nn.Module):
    """LeNet subsampling: average pool followed by per-map affine tanh."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, inputs: Tensor) -> Tensor:
        pooled = F.avg_pool2d(inputs, kernel_size=2, stride=2)
        return torch.tanh(pooled * self.scale[None, :, None, None] + self.bias[None, :, None, None])


class LeNet5(nn.Module):
    """LeNet-5-style classifier for 32x32 monochrome characters.

    C3 is fully connected to all C1 maps.  This deliberately differs from
    the sparse C3 connection table in LeCun et al. (1998), and maps directly
    to PyTorch's standard ``Conv2d`` implementation.
    """

    def __init__(self, num_classes: int = 62) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2 = TrainableAveragePool2d(6)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = TrainableAveragePool2d(16)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        x = torch.tanh(self.c1(inputs))
        x = self.s2(x)
        x = torch.tanh(self.c3(x))
        x = self.s4(x)
        x = torch.tanh(self.c5(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self.f6(x))
        return self.output(x)
