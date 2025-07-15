"""Sub-module containing utilities for various geometry operations."""

# needed to import for allowing type-hinting: torch.Tensor | np.ndarray
from __future__ import annotations

import torch
import torch.nn.functional

"""
General
"""


def generate_grid_positions(n: int, spacing: float) -> torch.Tensor:
    """
    Generates (x, y) positions for `n` robots in a centered grid layout.

    Parameters:
    - n (int): Number of robots.
    - spacing (float): Distance between robots in the grid.

    Returns:
    - positions (torch.Tensor): Tensor of shape (n, 2) containing (x, y) positions.
    """
    # Determine the closest square grid dimensions
    rows = int(torch.floor(torch.sqrt(torch.tensor(n, dtype=torch.float32))))
    cols = int(torch.ceil(torch.tensor(n, dtype=torch.float32) / rows))

    # Generate grid indices using PyTorch tensors
    y_indices, x_indices = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    grid_positions = torch.stack((x_indices.reshape(-1), y_indices.reshape(-1)), dim=-1).float() * spacing

    # Select only the first n positions
    positions = grid_positions[:n]

    # Compute the grid center offset
    grid_center = torch.tensor([(cols - 1) * spacing / 2, (rows - 1) * spacing / 2])

    # Center the positions around the origin
    positions -= grid_center

    return positions
