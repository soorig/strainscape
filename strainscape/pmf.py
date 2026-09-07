"""Strain -> gauge field, pseudo-magnetic field, sublattice polarization.

Conventions (x along the zigzag direction; A -> B bond vectors
delta_1 = a0 (sqrt3/2, 1/2), delta_2 = a0 (-sqrt3/2, 1/2), delta_3 = a0 (0, -1)):

    A      = (hbar/e) (beta / 2 a0) (eps_xx - eps_yy, -2 eps_xy)       [T m]
    B_ps   = d_x A_y - d_y A_x                                          [T]
    P      = sum_n (d_n / a0) delta_n = (3 a0 / 4) (2 eps_xy, eps_xx - eps_yy)   [m]
    B_ps   = -(hbar/e) (2 beta / 3 a0^2) div P

The last identity (Eq. 3 of Lu et al. 2023, up to the sign convention of P)
follows from the two lines above; it is checked numerically in the tests.
"""
from __future__ import annotations

import numpy as np

from .constants import A0, BETA, HBAR_OVER_E
from .elasticity import ddx, ddy


def gauge_field(exx, eyy, exy, beta: float = BETA, a0: float = A0):
    pref = HBAR_OVER_E * beta / (2.0 * a0)
    return pref * (exx - eyy), pref * (-2.0 * exy)


def pseudo_magnetic_field(exx, eyy, exy, dx: float, beta: float = BETA, a0: float = A0):
    Ax, Ay = gauge_field(exx, eyy, exy, beta, a0)
    return ddx(Ay, dx) - ddy(Ax, dx)


def sublattice_polarization(exx, eyy, exy, a0: float = A0):
    """P = sum_n (d_n/a0) delta_n with d_n = a0 (1 + n_hat . eps . n_hat)."""
    return 0.75 * a0 * 2.0 * exy, 0.75 * a0 * (exx - eyy)


def pmf_from_polarization(Px, Py, dx: float, beta: float = BETA, a0: float = A0):
    div = ddx(Px, dx) + ddy(Py, dx)
    return -HBAR_OVER_E * (2.0 * beta / (3.0 * a0 ** 2)) * div


def exciton_shift(exx, eyy, gauge_eV_per_strain: float = -5.0):
    """Strain-induced band-gap / exciton shift of a TMD monolayer on the same pillar [eV].

    Delta E ~ g * tr(eps), with g ~ -50 meV per % for monolayer WSe2 (order of magnitude;
    the deterministic single-photon emitters of Branny 2017 / Palacios-Berraquero 2017 and
    the strain-tuned emitters of Yu et al., Nano Lett. 2025 sit in the minima of this potential).
    """
    return gauge_eV_per_strain * (exx + eyy)


def uniform_field_from_triaxial(c: float, beta: float = BETA, a0: float = A0) -> float:
    """B_ps for the triaxial displacement u = c (2xy, x^2 - y^2)  (Guinea et al. 2010).

    eps_xx = 2cy, eps_yy = -2cy, eps_xy = 2cx  =>  A ∝ (4cy, -4cx)  =>  B = -8c * pref.
    """
    return -8.0 * c * HBAR_OVER_E * beta / (2.0 * a0)
