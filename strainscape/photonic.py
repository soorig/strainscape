"""Photonic twin: strain-induced pseudo-magnetic field in a coupled-waveguide array.

A honeycomb array of evanescently coupled waveguides obeys  i dpsi/dz = H psi
with H the coupling matrix (a weighted graph adjacency matrix); z plays the
role of time.  Displacing the waveguides with the same triaxial pattern that
gives graphene a uniform B_ps yields photonic pseudo-Landau levels
(cf. Rechtsman et al., Nat. Photon. 2013; Barczyk et al., Nat. Photon. 2024).

The coupling law kappa(d) = kappa0 exp(-(d - d0)/ell) is *calibrated* from a
finite-difference mode solver for two parallel slab waveguides, including a
mesh-convergence study of kappa itself.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal

from .tightbinding import Flake, honeycomb_flake, triaxial_displacement


def slab_pair_neff(gap: float, width: float = 450e-9, n_core: float = 3.48, n_clad: float = 1.444,
                   wavelength: float = 1.55e-6, dx: float = 5e-9, pad: float = 1.5e-6, n_modes: int = 2):
    """Effective indices of the two lowest TE supermodes of two parallel slabs (1D FD)."""
    k0 = 2 * np.pi / wavelength
    half = width + gap / 2 + pad
    x = np.arange(-half, half, dx)
    n = np.full_like(x, n_clad)
    n[(np.abs(x) > gap / 2) & (np.abs(x) < gap / 2 + width)] = n_core
    diag = -2.0 / dx ** 2 + (k0 * n) ** 2
    off = np.full(len(x) - 1, 1.0 / dx ** 2)
    beta2 = eigh_tridiagonal(diag, off, eigvals_only=True,
                             select="i", select_range=(len(x) - n_modes, len(x) - 1))
    return np.sqrt(beta2[::-1]) / k0     # descending: even, odd


def coupling_coefficient(gap: float, dx: float = 5e-9, **kw) -> float:
    """kappa = (beta_even - beta_odd) / 2  [rad/m]."""
    wavelength = kw.get("wavelength", 1.55e-6)
    neff = slab_pair_neff(gap, dx=dx, **kw)
    return 0.5 * (neff[0] - neff[1]) * 2 * np.pi / wavelength


def calibrate_coupling(gaps, dx_list, **kw):
    """kappa[gap, dx] table and an exponential fit on the finest mesh."""
    table = np.array([[coupling_coefficient(g, dx=dx, **kw) for dx in dx_list] for g in gaps])
    kappa_fine = table[:, -1]
    slope, intercept = np.polyfit(np.asarray(gaps), np.log(kappa_fine), 1)
    ell = -1.0 / slope
    kappa_at = lambda d: np.exp(intercept + slope * d)  # noqa: E731
    return table, ell, kappa_at


def photonic_lattice(radius_sites: float, d0: float, ell: float, kappa0: float, c: float = 0.0):
    """Honeycomb waveguide array; returns Flake (in metres), dense H [rad/m], beta_ph."""
    fl = honeycomb_flake(radius_sites * d0, a0=d0)
    u = triaxial_displacement(fl.pos, c) if c != 0 else None
    r = fl.pos + (u if u is not None else 0.0)
    i, j = fl.pairs[:, 0], fl.pairs[:, 1]
    d = np.linalg.norm(r[i] - r[j], axis=1)
    kap = kappa0 * np.exp(-(d - d0) / ell)
    H = np.zeros((fl.n, fl.n))
    H[i, j] = kap
    H[j, i] = kap
    beta_ph = d0 / ell          # -d ln kappa / d ln d at d0
    return fl, H, beta_ph


def photonic_pmf(c: float, d0: float, beta_ph: float) -> float:
    """Photonic pseudo-magnetic field (flux density, 1/m^2) for the triaxial pattern."""
    return -8.0 * c * beta_ph / (2.0 * d0)


def photonic_landau_levels(Bph: float, kappa0: float, d0: float, n_max: int = 3) -> np.ndarray:
    """E_n = sgn(n) v sqrt(2 |B| |n|),  v = 3 kappa0 d0 / 2   [rad/m]."""
    v = 1.5 * kappa0 * d0
    n = np.arange(-n_max, n_max + 1)
    return np.sign(n) * v * np.sqrt(2.0 * abs(Bph) * np.abs(n))


def spreading_radius(intensity: np.ndarray, pos: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """rms distance of the intensity distribution from ``centre`` for each z  [m]."""
    r2 = np.sum((pos - centre[None, :]) ** 2, axis=1)
    return np.sqrt(intensity @ r2 / intensity.sum(axis=1))


def propagate(H: np.ndarray, psi0: np.ndarray, z: np.ndarray) -> np.ndarray:
    """|psi(z)|^2 for the walk exp(-i H z); shape (len(z), n)."""
    vals, vecs = np.linalg.eigh(H)
    a = vecs.T @ psi0
    out = np.empty((len(z), H.shape[0]))
    for k, zz in enumerate(z):
        psi = vecs @ (np.exp(-1j * vals * zz) * a)
        out[k] = np.abs(psi) ** 2
    return out
