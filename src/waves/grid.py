"""Uniform Cartesian grid on the unit square."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Grid:
    """
    Uniform N x N grid on [0, 1]^2, including boundary nodes.

    Only the (N-2)^2 *interior* nodes are unknowns: homogeneous Dirichlet
    boundary conditions (u = 0) are imposed strongly by never storing or
    updating boundary values.
    """

    N: int
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    X: npt.NDArray[np.float64]
    Y: npt.NDArray[np.float64]
    h: float

    @property
    def m(self) -> int:
        """Number of interior nodes per axis (N - 2)."""
        return self.N - 2


def create_grid(N: int = 201) -> Grid:
    """Build a uniform N x N grid on [0, 1]^2 using 'ij' meshgrid indexing."""
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    h = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    return Grid(N=N, x=x, y=y, X=X, Y=Y, h=h)


def to_interior(field: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Flatten the interior of a 2D field to a vector of unknowns."""
    return field[1:-1, 1:-1].ravel()


def to_field(vec: npt.NDArray[np.float64], N: int) -> npt.NDArray[np.float64]:
    """Inverse of :func:`to_interior`: rebuild the full field, boundary = 0."""
    f = np.zeros((N, N))
    f[1:-1, 1:-1] = vec.reshape(N - 2, N - 2)
    return f
