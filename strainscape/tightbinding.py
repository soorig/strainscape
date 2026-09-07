"""Tight-binding honeycomb lattice with strain-modified hoppings.

The Hamiltonian is a weighted graph adjacency matrix on the honeycomb graph:
    H_ij = -t0 exp(-beta (d_ij / a0 - 1))      for nearest neighbours,
so pseudo-Landau levels are spectral clustering of a graph operator and the
continuous-time quantum walk exp(-iHt) is its dynamics.  Used to (i) validate
the continuum B_ps against exact Landau-level positions on a uniformly strained
flake and (ii) expose the sublattice polarisation of the zeroth pseudo-Landau
level that underlies the SHG mechanism of Lu et al. 2023.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from .constants import A0, BETA, E_CHARGE, HBAR, T0, V_F

DELTA = np.array([[np.sqrt(3) / 2, 0.5], [-np.sqrt(3) / 2, 0.5], [0.0, -1.0]])  # A->B bonds / a0


@dataclass
class Flake:
    pos: np.ndarray      # (n, 2) undeformed positions [m]
    sub: np.ndarray      # (n,) 0 = A, 1 = B
    pairs: np.ndarray    # (m, 2) nearest-neighbour index pairs
    a0: float = A0

    @property
    def n(self) -> int:
        return self.pos.shape[0]


def honeycomb_flake(radius: float, a0: float = A0) -> Flake:
    """Disc-shaped honeycomb flake, zigzag direction along x."""
    a = a0 * np.sqrt(3.0)
    a1 = a * np.array([1.0, 0.0])
    a2 = a * np.array([0.5, np.sqrt(3.0) / 2])
    m = int(np.ceil(radius / a)) + 2
    n1, n2 = np.meshgrid(np.arange(-2 * m, 2 * m + 1), np.arange(-2 * m, 2 * m + 1))
    n1, n2 = n1.ravel(), n2.ravel()
    A = n1[:, None] * a1[None, :] + n2[:, None] * a2[None, :]
    B = A + a0 * DELTA[2][None, :]
    pos = np.vstack([A, B])
    sub = np.concatenate([np.zeros(len(A), dtype=int), np.ones(len(B), dtype=int)])
    keep = np.hypot(pos[:, 0], pos[:, 1]) <= radius
    pos, sub = pos[keep], sub[keep]
    tree = cKDTree(pos)
    pairs = np.array(sorted(tree.query_pairs(r=1.1 * a0)), dtype=int)
    # drop dangling atoms (coordination 1) once, for a cleaner edge
    deg = np.bincount(pairs.ravel(), minlength=len(pos))
    good = deg >= 2
    if not good.all():
        remap = -np.ones(len(pos), dtype=int)
        remap[good] = np.arange(good.sum())
        pos, sub = pos[good], sub[good]
        pairs = remap[pairs]
        pairs = pairs[(pairs >= 0).all(axis=1)]
    return Flake(pos, sub, pairs, a0)


def triaxial_displacement(pos: np.ndarray, c: float) -> np.ndarray:
    """u = c (2xy, x^2 - y^2): uniform pseudo-magnetic field (Guinea et al. 2010)."""
    x, y = pos[:, 0], pos[:, 1]
    return np.column_stack([2.0 * c * x * y, c * (x ** 2 - y ** 2)])


def hamiltonian(flake: Flake, displacement: np.ndarray | None = None,
                height: np.ndarray | None = None, t0: float = T0, beta: float = BETA) -> sp.csr_matrix:
    """Sparse H [eV] with hoppings evaluated on the deformed 3D positions."""
    r = flake.pos.copy()
    if displacement is not None:
        r = r + displacement
    i, j = flake.pairs[:, 0], flake.pairs[:, 1]
    d2 = np.sum((r[i] - r[j]) ** 2, axis=1)
    if height is not None:
        d2 = d2 + (height[i] - height[j]) ** 2
    d = np.sqrt(d2)
    t = -t0 * np.exp(-beta * (d / flake.a0 - 1.0))
    H = sp.coo_matrix((np.concatenate([t, t]), (np.concatenate([i, j]), np.concatenate([j, i]))),
                      shape=(flake.n, flake.n))
    return H.tocsr()


def spectrum_near(H: sp.csr_matrix, k: int = 80, sigma: float = 1.3e-4):
    """k eigenpairs closest to ``sigma`` (shift-invert Lanczos).

    ``sigma`` is kept slightly off zero: finite flakes carry exact zero modes
    (sublattice imbalance at the edge) that make the shifted matrix singular.
    """
    vals, vecs = spla.eigsh(H.tocsc(), k=k, sigma=sigma, which="LM")
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def landau_level_energies(B: float, n_max: int = 4, v_f: float = V_F) -> np.ndarray:
    """E_n = sgn(n) v_F sqrt(2 e hbar |B| |n|)  [eV], n = -n_max..n_max."""
    n = np.arange(-n_max, n_max + 1)
    return np.sign(n) * v_f * np.sqrt(2.0 * E_CHARGE * HBAR * abs(B) * np.abs(n)) / E_CHARGE


def magnetic_length(B: float) -> float:
    return np.sqrt(HBAR / (E_CHARGE * abs(B)))


def dos(vals: np.ndarray, energies: np.ndarray, eta: float) -> np.ndarray:
    """Lorentzian-broadened density of states."""
    return np.sum(eta / np.pi / ((energies[:, None] - vals[None, :]) ** 2 + eta ** 2), axis=1)


def sublattice_weight(vecs: np.ndarray, vals: np.ndarray, sub: np.ndarray, window: float,
                      interior: np.ndarray | None = None):
    """Weight of states with |E| < window on A and on B sites (interior only)."""
    sel = np.abs(vals) < window
    w = np.sum(np.abs(vecs[:, sel]) ** 2, axis=1)
    if interior is not None:
        w = np.where(interior, w, 0.0)
    return w[sub == 0].sum(), w[sub == 1].sum()


def average_mixing_diag(H_dense: np.ndarray) -> np.ndarray:
    """Diagonal of the average mixing matrix  M = sum_r E_r o E_r  (return probabilities).

    Uses the eigenspace projectors E_r of H; degenerate eigenvalues are grouped.
    """
    vals, vecs = np.linalg.eigh(H_dense)
    n = len(vals)
    diag = np.zeros(n)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(vals[j] - vals[i]) < 1e-9:
            j += 1
        V = vecs[:, i:j]
        diag += np.sum(V * V, axis=1) ** 2  # (E_r)_{uu}^2
        i = j
    return diag
