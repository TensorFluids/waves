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


## Requirements

Python 3.9 or newer is recommended. Install the Python dependencies with:

```bash
python -m pip install numpy scipy matplotlib jupyter
```

To export an MP4 animation, install [FFmpeg](https://ffmpeg.org/) and make
sure it is available to Matplotlib. The first notebook cell currently contains
an example Windows-specific FFmpeg path; update it for your local installation
or replace it with the path to your own `ffmpeg` executable.

## Running the solver

1. Clone the repository and enter the project directory:

   ```bash
   git clone https://github.com/<your-username>/waves.git
   cd waves
   ```

2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open [`Wave Equ.ipynb`](./Wave%20Equ.ipynb) and run the cells in order.

The final cell runs the viscous simulation, plots normalized energy
`E(t) / E(0)`, and writes the animation to
`wave_carpet_viscous.mp4`. If FFmpeg is unavailable, the animation helper falls
back to a GIF using Pillow.

## Experimenting

The main parameters can be changed in the final `simulate(...)` call:

```python
X, Y, frames, times, energies = simulate(
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

## Project status

This repository is intentionally minimal and notebook-based. Possible future
extensions include separating the solver into a Python module, adding automated
convergence tests, supporting alternative boundary conditions, and comparing
against analytical solutions or explicit time-stepping schemes.
