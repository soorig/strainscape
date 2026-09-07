"""End-to-end forward model: pillar geometry -> drape -> strain -> B_ps, P."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .constants import RAMAN_FWHM
from .elasticity import equilibrium_residual, strain_from_height
from .pmf import pseudo_magnetic_field, sublattice_polarization
from .topography import Grid, Pillar, drape_height, tent_mask


@dataclass
class ForwardResult:
    grid: Grid
    pillar: Pillar
    rounding: float
    h: np.ndarray
    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    B: np.ndarray
    Px: np.ndarray
    Py: np.ndarray

    @property
    def principal_strain(self) -> np.ndarray:
        m = 0.5 * (self.exx + self.eyy)
        r = np.sqrt((0.5 * (self.exx - self.eyy)) ** 2 + self.exy ** 2)
        return m + r

    @property
    def P_mag(self) -> np.ndarray:
        return np.hypot(self.Px, self.Py)

    def tent(self, margin: float = 0.0) -> np.ndarray:
        return tent_mask(self.pillar, self.grid, margin)


def run_forward(pillar: Pillar, grid: Grid, rounding: float = 0.0, relax: str = "pinned",
                h: np.ndarray | None = None) -> ForwardResult:
    """Forward model.

    relax="pinned" (default): slack-fed geometric strain eps = 1/2 grad h grad h^T; the
    sheet adheres where it lands and the flat floor stays unstrained, as in the Raman
    maps of Kang 2021 / Lu 2023.  relax="free": frictionless in-plane relaxation inside
    the periodic cell (lower bound on strain inhomogeneity).
    """
    if h is None:
        h = drape_height(pillar, grid, rounding)
    exx, eyy, exy, ux, uy = strain_from_height(h, grid.dx, relax=relax)
    B = pseudo_magnetic_field(exx, eyy, exy, grid.dx)
    Px, Py = sublattice_polarization(exx, eyy, exy)
    return ForwardResult(grid, pillar, rounding, h, exx, eyy, exy, ux, uy, B, Px, Py)


def raman_convolve(field: np.ndarray, dx: float, fwhm: float = RAMAN_FWHM) -> np.ndarray:
    """Convolve with the diffraction-limited Raman spot (Gaussian, FWHM 361 nm)."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) / dx
    return gaussian_filter(field, sigma=sigma, mode="wrap")


def summarize(res: ForwardResult, B_thr: float = 20.0) -> dict:
    eps = res.principal_strain
    tent = res.tent(margin=2 * res.grid.dx)
    absB = np.abs(res.B)
    return {
        "dx_nm": res.grid.dx * 1e9,
        "N": res.grid.N,
        "eps_max_pct": 100.0 * eps.max(),
        "eps_raman_max_pct": 100.0 * raman_convolve(eps, res.grid.dx).max(),
        "B_max_T": absB.max(),
        "B_rms_tent_T": float(np.sqrt(np.mean(res.B[tent] ** 2))) if tent.any() else 0.0,
        "B_mean_abs_tent_T": float(np.mean(absB[tent])) if tent.any() else 0.0,
        "coverage_frac": float(np.mean(absB >= B_thr)),
        "coverage_area_um2": float(np.mean(absB >= B_thr)) * (res.grid.L * 1e6) ** 2,
        "equilibrium_residual": equilibrium_residual(res.exx, res.eyy, res.exy, res.grid.dx),
    }
