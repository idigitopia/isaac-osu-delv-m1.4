import torch
import torch.nn as nn


class ScaleLayer(nn.Module):
    """
    Simple scaling layer definition, just multiply output by a fixed scaling factor. Intended to be
    used for bounded actors.
    """

    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, x):
        return x * self.scale
