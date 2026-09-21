"""Initial condition generators."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from waves.grid import Grid


def gaussian_bump(
    grid: Grid,
    amplitude: float = 1.0,
    width: float = 100.0,
    center: tuple[float, float] = (0.5, 0.5),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Gaussian bump displacement, zero initial velocity:

        u(x, y, 0) = amplitude * exp(-width * ((x-cx)^2 + (y-cy)^2))
        v(x, y, 0) = 0

    Boundary values are forced to zero (strong Dirichlet BC).
    """
    cx, cy = center
    X, Y = grid.X, grid.Y
    u0 = amplitude * np.exp(-width * ((X - cx) ** 2 + (Y - cy) ** 2))
    u0[0, :] = u0[-1, :] = u0[:, 0] = u0[:, -1] = 0.0
    v0 = np.zeros_like(u0)
    return u0, v0
