"""Physics/energy sanity checks for the waves solver."""

from __future__ import annotations

import numpy as np

from waves.solver import simulate


def test_energy_is_conserved_without_damping():
    """With nu = gamma = 0, discrete energy should stay ~constant."""
    result = simulate(N=41, T=0.5, cfl=0.4, nu=0.0, gamma=0.0, verbose=False)
    E0 = result.energies[0]
    relative_drift = np.abs(result.energies - E0).max() / E0
    assert relative_drift < 1e-8


def test_energy_decays_monotonically_with_viscosity():
    """With nu > 0, discrete energy must never increase between stored frames."""
    result = simulate(N=41, T=0.5, cfl=0.4, nu=2e-3, gamma=0.0, verbose=False)
    diffs = np.diff(result.energies)
    assert np.all(diffs <= 1e-10)
    assert result.energies[-1] < result.energies[0]


def test_energy_decays_with_linear_drag():
    """With gamma > 0 and nu = 0, energy must also decay monotonically."""
    result = simulate(N=41, T=0.5, cfl=0.4, nu=0.0, gamma=0.5, verbose=False)
    diffs = np.diff(result.energies)
    assert np.all(diffs <= 1e-10)
    assert result.energies[-1] < result.energies[0]
