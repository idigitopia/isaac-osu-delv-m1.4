import torch.nn as nn


class DepthMapEncoder(nn.Module):
    def __init__(self, in_channels: int, encoded_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, encoded_dim),
        )

    def forward(self, x):
        # Check if it has sequence dimension
        if x.dim() == 5:
            B, N = x.shape[:2]
            x = x.flatten(0, 1)
            x = self.encoder(x)
            x = x.view(B, N, -1)
        else:
            x = self.encoder(x)

        return x  # noqa: R504
