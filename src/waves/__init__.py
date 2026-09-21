"""
waves
=====

A small package for solving the 2D (linear, optionally viscously damped)
wave equation on a unit square with homogeneous Dirichlet boundary
conditions, using finite differences in space and an implicit theta-method
in time.

See :func:`waves.solver.simulate` for the main entry point.
"""

from waves.solver import simulate

__all__ = ["simulate"]

__version__ = "0.1.0"
