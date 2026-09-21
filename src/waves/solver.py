"""Top-level simulation driver."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from waves.energy import energy
from waves.grid import Grid, create_grid, to_field, to_interior
from waves.initial_conditions import gaussian_bump
from waves.integrators import ThetaIntegrator
from waves.operators import build_A, laplacian


@dataclass
class SimulationResult:
    """Container for the outputs of :func:`simulate`."""

    grid: Grid
    frames: list[npt.NDArray[np.float64]]
    times: list[float]
    energies: npt.NDArray[np.float64]
    dt: float
    nsteps: int
    params: dict = field(default_factory=dict)

    @property
    def X(self) -> npt.NDArray[np.float64]:
        return self.grid.X

    @property
    def Y(self) -> npt.NDArray[np.float64]:
        return self.grid.Y


def simulate(
    N: int = 201,
    c: float = 1.0,
    T: float = 1.5,
    cfl: float = 0.4,
    store_every: int = 4,
    nu: float = 0.0,
    gamma: float = 0.0,
    theta: float = 0.5,
    verbose: bool = True,
) -> SimulationResult:
    """
    Simulate the (optionally viscously damped) 2D wave equation

        u_tt = c^2 (u_xx + u_yy) + nu * (u_xxt + u_yyt) - gamma * u_t

    on [0, 1]^2 with homogeneous Dirichlet boundary conditions, starting
    from a centered Gaussian bump with zero initial velocity.

    Parameters
    ----------
    N : number of grid points per axis (including boundary).
    c : wave speed.
    T : total simulation time.
    cfl : sets dt = cfl * h / c. Since the theta-method (theta >= 1/2) is
        unconditionally stable, this controls accuracy, not stability.
    store_every : number of steps between stored frames (for animation).
    nu : Kelvin-Voigt viscosity (damps high-frequency modes ~ nu * k^2).
    gamma : linear drag (damps all modes uniformly).
    theta : time-integration parameter (0.5 = Crank-Nicolson, 1 = backward Euler).
    verbose : print a short diagnostic summary at the end of the run.

    Returns
    -------
    SimulationResult with the recorded frames, times, and discrete energy
    history (``E(t)``), useful for verifying conservation (nu = gamma = 0)
    or monotone decay (nu > 0 or gamma > 0).
    """
    grid = create_grid(N)
    dt = cfl * grid.h / c  # accuracy-driven; theta >= 1/2 has no stability limit
    nsteps = int(round(T / dt))

    L = laplacian(N, grid.h)
    A = build_A(L, c, nu=nu, gamma=gamma)
    n = A.shape[0]

    integrator = ThetaIntegrator(A, dt, theta=theta)

    u0, v0 = gaussian_bump(grid)
    Yn = np.concatenate([to_interior(u0), to_interior(v0)])

    E0 = energy(Yn, L, c)
    E_prev = E0
    max_increase = 0.0  # should stay ~0: energy must never grow

    frames = [to_field(Yn[: n // 2], N)]
    times = [0.0]
    energies = [E0]

    for step in range(1, nsteps + 1):
        Yn = integrator.step(Yn)
        E = energy(Yn, L, c)
        max_increase = max(max_increase, E - E_prev)
        E_prev = E
        if step % store_every == 0:
            frames.append(to_field(Yn[: n // 2], N))
            times.append(step * dt)
            energies.append(E)

    if verbose:
        stiff = 8.0 * nu * dt / grid.h**2  # viscous stiffness indicator
        print(f"N={N}, dt={dt:.5f}, steps={nsteps}, frames={len(frames)}")
        print(
            f"nu={nu:g}, gamma={gamma:g}, theta={theta:g}, "
            f"viscous stiffness dt*nu*|L| = {stiff:.2f}"
        )
        print(
            f"energy: E0={E0:.6e} -> E_end={E_prev:.6e}  "
            f"({100.0 * (1.0 - E_prev / E0):.2f}% dissipated)"
        )
        print(
            f"max per-step energy INCREASE: {max_increase / E0:.2e} "
            "(must be ~0 for a dissipative system)"
        )

    return SimulationResult(
        grid=grid,
        frames=frames,
        times=times,
        energies=np.array(energies),
        dt=dt,
        nsteps=nsteps,
        params=dict(N=N, c=c, T=T, cfl=cfl, nu=nu, gamma=gamma, theta=theta),
    )
