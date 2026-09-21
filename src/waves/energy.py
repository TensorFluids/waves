"""Discrete energy diagnostics used to verify solver correctness."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp


def energy(Y: npt.NDArray[np.float64], L: sp.csr_matrix, c: float) -> float:
    """
    Discrete energy of the state Y = [u, v]:

        E = 1/2 ( ||v||^2 - c^2 u^T L u )

    -L is symmetric positive definite, so E >= 0. With no damping
    (nu = gamma = 0) this is exactly conserved by the theta = 1/2
    (Crank-Nicolson) scheme; with damping it must decrease monotonically.
    """
    n = Y.size // 2
    u, v = Y[:n], Y[n:]
    return 0.5 * (v @ v - c**2 * (u @ (L @ u)))


def dissipation_rate(
    Y: npt.NDArray[np.float64],
    L: sp.csr_matrix,
    nu: float,
    gamma: float,
) -> float:
    """
    Continuous-level energy budget:

        dE/dt = nu * v^T L v - gamma * ||v||^2  <= 0

    since L is negative definite. Used to check that the *measured* decay
    matches the physics rather than coming from numerical damping.
    """
    n = Y.size // 2
    v = Y[n:]
    return nu * (v @ (L @ v)) - gamma * (v @ v)
