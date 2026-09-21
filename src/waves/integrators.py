"""Implicit time integrators for the linear first-order system Y_t = A Y."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class ThetaIntegrator:
    """
    Generalised trapezoidal / theta-method time-stepper for a constant,
    linear system Y_t = A Y:

        (I - theta * dt * A) Y_{n+1} = (I + (1 - theta) * dt * A) Y_n

    theta = 1/2 -> Crank-Nicolson (2nd order, A-stable, non-dissipative)
    theta = 1   -> backward Euler (1st order, L-stable)

    The system matrix is constant in time, so the LHS is assembled and
    LU-factorized once in ``__post_init__`` and reused at every step.
    """

    A: sp.spmatrix
    dt: float
    theta: float = 0.5

    def __post_init__(self) -> None:
        n = self.A.shape[0]
        I = sp.identity(n, format="csr")
        LHS = (I - self.theta * self.dt * self.A).tocsc()
        self._RHS = (I + (1.0 - self.theta) * self.dt * self.A).tocsr()
        self._solve: Callable[[npt.NDArray], npt.NDArray] = spla.factorized(LHS)

    def step(self, Y: npt.NDArray) -> npt.NDArray:
        """Advance the state vector Y by one time step of size ``dt``."""
        return self._solve(self._RHS @ Y)
