"""Membrane strain of graphene draped over a topography.

Small-slope Foeppl-von Karman kinematics with a prescribed out-of-plane
profile h(x, y):

    eps_ij = 1/2 (d_i u_j + d_j u_i) + 1/2 d_i h d_j h

The in-plane displacement u relaxes to mechanical equilibrium of an isotropic
plane-stress sheet (``relax="free"``: frictionless sliding inside the periodic
cell) or is frozen (``relax="pinned"``: sheet glued to the substrate, u = 0).
The equilibrium problem is solved exactly on the periodic grid by FFT using the
central-difference symbol K_j = sin(k_j dx)/dx, so that the strain returned is
consistent with the finite-difference gradient used everywhere else.
"""
from __future__ import annotations

import numpy as np

from .constants import LAMBDA_2D, MU_2D


def ddx(f: np.ndarray, dx: float) -> np.ndarray:
    """Central difference along x (axis 1), periodic."""
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)


def ddy(f: np.ndarray, dx: float) -> np.ndarray:
    """Central difference along y (axis 0), periodic."""
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dx)


def geometric_strain(h: np.ndarray, dx: float):
    hx, hy = ddx(h, dx), ddy(h, dx)
    return 0.5 * hx * hx, 0.5 * hy * hy, 0.5 * hx * hy


def _symbols(N: int, dx: float):
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    Kx = np.sin(k * dx) / dx           # central-difference symbol (D f)^ = i K f^
    KX, KY = np.meshgrid(Kx, Kx)
    return KX, KY


def relax_inplane(gxx: np.ndarray, gyy: np.ndarray, gxy: np.ndarray, dx: float,
                  mu: float = MU_2D, lam: float = LAMBDA_2D):
    """Return in-plane displacement (ux, uy) minimising the membrane energy.

    Solves  mu D^2 u_i + (lam+mu) D_i D_j u_j = -(lam D_i g_kk + 2 mu D_j g_ij).
    """
    N = gxx.shape[0]
    KX, KY = _symbols(N, dx)
    gkk_hat = np.fft.fft2(gxx + gyy)
    gxx_hat, gyy_hat, gxy_hat = np.fft.fft2(gxx), np.fft.fft2(gyy), np.fft.fft2(gxy)
    # r_i = i (lam K_i g_kk^ + 2 mu K_j g_ij^)
    rx = 1j * (lam * KX * gkk_hat + 2.0 * mu * (KX * gxx_hat + KY * gxy_hat))
    ry = 1j * (lam * KY * gkk_hat + 2.0 * mu * (KX * gxy_hat + KY * gyy_hat))
    K2 = KX ** 2 + KY ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(K2 > 0, 1.0 / (mu * K2), 0.0)
        fac = np.where(K2 > 0, (lam + mu) / ((lam + 2.0 * mu) * K2), 0.0)
    kr = KX * rx + KY * ry
    ux_hat = inv * (rx - fac * KX * kr)
    uy_hat = inv * (ry - fac * KY * kr)
    ux = np.real(np.fft.ifft2(ux_hat))
    uy = np.real(np.fft.ifft2(uy_hat))
    return ux, uy


def strain_from_height(h: np.ndarray, dx: float, relax: str = "free"):
    """Total membrane strain (eps_xx, eps_yy, eps_xy) and displacement (ux, uy)."""
    gxx, gyy, gxy = geometric_strain(h, dx)
    if relax == "pinned":
        z = np.zeros_like(h)
        return gxx, gyy, gxy, z, z
    if relax != "free":
        raise ValueError("relax must be 'free' or 'pinned'")
    ux, uy = relax_inplane(gxx, gyy, gxy, dx)
    exx = ddx(ux, dx) + gxx
    eyy = ddy(uy, dx) + gyy
    exy = 0.5 * (ddy(ux, dx) + ddx(uy, dx)) + gxy
    return exx, eyy, exy, ux, uy


def equilibrium_residual(exx, eyy, exy, dx, mu: float = MU_2D, lam: float = LAMBDA_2D):
    """|div sigma| / |sigma| for the plane-stress sheet; ~0 at equilibrium."""
    tr = exx + eyy
    sxx = lam * tr + 2 * mu * exx
    syy = lam * tr + 2 * mu * eyy
    sxy = 2 * mu * exy
    fx = ddx(sxx, dx) + ddy(sxy, dx)
    fy = ddx(sxy, dx) + ddy(syy, dx)
    scale = np.sqrt(np.mean(sxx ** 2 + syy ** 2 + 2 * sxy ** 2)) / dx
    return np.sqrt(np.mean(fx ** 2 + fy ** 2)) / scale
