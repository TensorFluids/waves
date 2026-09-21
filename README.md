# Waves 🌊

A small numerical experiment for solving the two-dimensional wave equation on a
unit square. The project uses finite differences and sparse linear algebra to
simulate wave propagation, investigate viscous damping, monitor the discrete
energy, and generate a rotating 3D surface animation.


## Model

The basic problem is

```text
u_tt = c² (u_xx + u_yy),     (x, y) ∈ [0, 1]²
u = 0                         on the boundary
```

The notebook also supports a damped/viscous extension:

```text
u_tt = c² (u_xx + u_yy)
       + ν (u_xxt + u_yyt)
       - γ u_t
```

Here, `ν` is a Kelvin–Voigt-style viscous damping coefficient that damps
short wavelengths more strongly, while `γ` is uniform velocity drag.

## Numerical method

- Uniform `N × N` grid with homogeneous Dirichlet boundary conditions.
- Second-order central-difference Laplacian on the interior grid points.
- Sparse Kronecker-sum construction of the discrete Laplacian.
- First-order formulation with displacement `u` and velocity `v`.
- Implicit theta-method time integration:

  ```text
  (I - θ Δt A) Yⁿ⁺¹ = (I + (1 - θ) Δt A) Yⁿ
  ```

- `θ = 1/2` gives the second-order Crank–Nicolson/trapezoidal rule.
- The constant sparse system matrix is LU-factorized once and reused at every
  time step.
- A discrete energy is tracked to verify conservation in the undamped case
  and monotone decay when damping is enabled.

For the default viscous run, the notebook uses:

```text
N = 201,  c = 1.0,  T = 10,  ν = 2×10⁻⁴,  γ = 0,  θ = 0.5
```

The initial displacement is a Gaussian bump centered in the domain, with zero
initial velocity.


## Repository layout

```text
waves/
├── src/waves/            # installable package — all solver logic lives here
│   ├── grid.py           # uniform grid + interior/full-field conversions
│   ├── operators.py      # discrete Laplacian, first-order system matrix A
│   ├── initial_conditions.py
│   ├── integrators.py    # ThetaIntegrator (Crank-Nicolson / backward Euler)
│   ├── energy.py         # discrete energy + dissipation-rate diagnostics
│   ├── solver.py         # simulate(...) driver, returns a SimulationResult
│   └── visualization.py  # 3D "moving carpet" animation
├── notebooks/
│   └── Wave Equ.ipynb    # thin experimentation notebook, imports `waves`
├── tests/                # pytest suite (energy conservation, convergence, ...)
├── wave_carpet_viscous.mp4
└── pyproject.toml
```

The package (`src/waves`) is the source of truth for the numerics — it is
unit-tested and reusable. The notebook is kept deliberately thin: it just
calls into `waves` for parameter exploration, plotting, and generating
animations, so there is no numerics logic duplicated between the two.

## Requirements

Python 3.9 or newer is recommended.

Install the package (editable mode) with development and notebook
dependencies:

```bash
python -m pip install -e ".[dev]"
```

This pulls in `numpy`, `scipy`, `matplotlib`, `pytest`, and `jupyter`.

To export an MP4 animation, install [FFmpeg](https://ffmpeg.org/) and make
sure it is available to Matplotlib (see the commented-out `ffmpeg_path` line
near the top of the notebook). If FFmpeg is unavailable, the animation
helper automatically falls back to a GIF via Pillow.

## Running the solver

From Python or a REPL:

```python
from waves.solver import simulate
from waves.visualization import animate_solution

result = simulate(N=201, c=1.0, T=10, cfl=0.4, nu=2e-4, gamma=0.0, theta=0.5)
animate_solution(result.X, result.Y, result.frames, result.times,
                  filename="wave_carpet_viscous.mp4")
```

Or interactively via the notebook:

```bash
jupyter notebook notebooks/"Wave Equ.ipynb"
```

`simulate(...)` returns a `SimulationResult` with `.frames`, `.times`,
`.energies`, `.X`, `.Y`, `.dt`, and `.nsteps` — everything needed for
plotting, animation, or further post-processing.

## Experimenting

The main parameters can be passed directly to `simulate(...)`:

```python
result = simulate(
    N=201,
    c=1.0,
    T=10,
    cfl=0.4,
    store_every=4,
    nu=2e-4,
    gamma=0.0,
    theta=0.5,
)
```

Useful experiments include:

- Set `nu=0` and `gamma=0` to recover the conservative wave equation.
- Increase `gamma` to apply uniform damping to all modes.
- Increase `nu` to damp high-frequency modes more strongly.
- Set `theta=1` to use backward Euler.
- Change `N` to study spatial resolution and computational cost.
- Change `store_every` to control the number of stored animation frames.

Although implicit methods do not impose the usual explicit CFL stability
restriction, the time step still affects accuracy and the fidelity of viscous
damping.

## Testing

```bash
python -m pytest tests/ -v
```

The test suite checks:

- Grid/operator correctness (spacing, symmetry, negative-definiteness of the
  discrete Laplacian).
- Exact energy conservation for the undamped case.
- Monotonic energy decay when viscosity (`nu`) or drag (`gamma`) is enabled.
- Relative behavior of Crank-Nicolson vs. backward Euler damping.
- A coarse convergence regression guard on the time step.

## Project status

The core solver has been refactored into a tested, importable package
(`src/waves`), with the original notebook kept as a lightweight
experimentation front-end. Possible future extensions include higher-order
spatial discretizations, absorbing/non-reflecting boundary conditions,
variable wave speed (heterogeneous media), iterative/GPU-accelerated linear
solvers, and comparisons against analytical solutions.
