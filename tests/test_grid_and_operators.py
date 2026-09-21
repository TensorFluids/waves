"""Structural/unit checks for grid and operator construction."""

from __future__ import annotations

import numpy as np

from waves.grid import create_grid, to_field, to_interior
from waves.operators import build_A, laplacian


def test_grid_spacing_and_shape():
    grid = create_grid(N=11)
    assert grid.X.shape == (11, 11)
    assert grid.Y.shape == (11, 11)
    assert np.isclose(grid.h, 0.1)
    assert grid.m == 9


def test_to_interior_and_to_field_are_inverses():
    grid = create_grid(N=11)
    field = np.random.default_rng(0).normal(size=(11, 11))
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = 0.0
    vec = to_interior(field)
    rebuilt = to_field(vec, 11)
    assert np.allclose(field, rebuilt)


def test_laplacian_is_symmetric_negative_definite():
    grid = create_grid(N=11)
    L = laplacian(grid.N, grid.h)
    dense = L.toarray()
    assert np.allclose(dense, dense.T)
    eigvals = np.linalg.eigvalsh(dense)
    assert np.all(eigvals < 0)


def test_build_A_shape_and_zero_damping_blocks():
    grid = create_grid(N=11)
    L = laplacian(grid.N, grid.h)
    A = build_A(L, c=1.0, nu=0.0, gamma=0.0)
    m2 = grid.m**2
    assert A.shape == (2 * m2, 2 * m2)
