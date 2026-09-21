"""Discrete spatial operators: the Laplacian and the first-order system matrix."""

from __future__ import annotations

import scipy.sparse as sp


def laplacian(N: int, h: float) -> sp.csr_matrix:
    """
    Second-order central-difference Laplacian on the interior nodes,
    built by Kronecker sums of the 1D operator

        D2 = (1/h^2) tridiag(1, -2, 1)

    The Dirichlet ghost contributions vanish (u = 0), so no extra terms
    appear on the right-hand side.

        L = D2 (x) I  +  I (x) D2      ("(x)" = Kronecker product)

    Returns an (m^2 x m^2) CSR matrix with m = N - 2, using 'ij' (row-major
    over x) ordering consistent with :func:`waves.grid.create_grid`.
    """
    m = N - 2
    diagonals = [
        [1.0] * (m - 1),
        [-2.0] * m,
        [1.0] * (m - 1),
    ]
    D2 = sp.diags(diagonals, [-1, 0, 1], format="csr") / h**2
    I = sp.identity(m, format="csr")
    return sp.kron(D2, I, format="csr") + sp.kron(I, D2, format="csr")


def build_A(
    L: sp.csr_matrix,
    c: float,
    nu: float = 0.0,
    gamma: float = 0.0,
) -> sp.csr_matrix:
    """
    Block operator of the first-order system  Y_t = A Y,  Y = [u, v]^T:

        A = [[ 0      I              ],
             [ c^2 L  nu L - gamma I ]]

    nu    : Kelvin-Voigt (strain-rate) viscosity, damps like nu * k^2
    gamma : linear drag, damps all modes uniformly

    Setting nu = gamma = 0 recovers the conservative (non-dissipative)
    wave equation.
    """
    n = L.shape[0]
    Z = sp.csr_matrix((n, n))
    I = sp.identity(n, format="csr")
    D = (nu * L - gamma * I).tocsr()  # velocity-damping block
    return sp.bmat([[Z, I], [c**2 * L, D]], format="csr")
