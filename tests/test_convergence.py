"""Empirical convergence check for the spatial/temporal discretization."""

from __future__ import annotations

import numpy as np

from waves.solver import simulate


def test_energy_drift_decreases_with_finer_time_step():
    """
    Smaller cfl (=> smaller dt) should not increase the relative energy
    drift for the conservative (nu = gamma = 0) Crank-Nicolson scheme.
    This is a coarse regression guard, not a formal order-of-accuracy test.
    """
    coarse = simulate(N=41, T=0.5, cfl=0.8, nu=0.0, gamma=0.0, verbose=False)
    fine = simulate(N=41, T=0.5, cfl=0.2, nu=0.0, gamma=0.0, verbose=False)

    drift_coarse = np.abs(coarse.energies - coarse.energies[0]).max() / coarse.energies[0]
    drift_fine = np.abs(fine.energies - fine.energies[0]).max() / fine.energies[0]

    assert drift_fine <= drift_coarse + 1e-12
