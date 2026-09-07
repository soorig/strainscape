"""Nanopillar topography and graphene drape profile.

The unit cell is a square, periodic cell of pitch ``L`` (one pillar per cell,
as in the 1 um pitch arrays of Kang 2021 / Lu 2023).  A pillar footprint is a
rounded (super-elliptic) polygon whose radius may be modulated by low-order
angular harmonics.  The graphene sheet is pinned to the pillar top (height H),
adhered to the substrate beyond a "tent width" w from the footprint, and in
between it takes a tension-dominated membrane shape (Laplace equation).  A
Gaussian rounding of radius ``rounding`` represents the bending length
sqrt(kappa/T) ~ 1 nm that smooths the kink at the pillar edge (the resolution
that the mesh has to be calibrated against).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import distance_transform_edt, gaussian_filter


@dataclass(frozen=True)
class Grid:
    """Square periodic cell of side ``L`` sampled on an ``N x N`` grid."""

    L: float
    N: int

    @property
    def dx(self) -> float:
        return self.L / self.N

    def axes(self):
        x = (np.arange(self.N) - self.N / 2) * self.dx
        return x, x.copy()

    def mesh(self):
        x, y = self.axes()
        return np.meshgrid(x, y)  # X[iy, ix], Y[iy, ix]


@dataclass
class Pillar:
    """Fabricable footprint description.

    half_side : half of the footprint side [m]
    power     : super-ellipse exponent (2 = disc, large = square with sharp corners)
    height    : pillar height [m]
    tent_width: distance from footprint edge to the adhered floor [m]
    rotation  : rotation of the footprint [rad]
    harmonics : {k: (a_k, b_k)} radial modulation rho(theta) = 1 + sum a_k cos k th + b_k sin k th
    """

    half_side: float = 250e-9
    power: float = 8.0
    height: float = 85e-9
    tent_width: float = 150e-9
    rotation: float = 0.0
    harmonics: dict = field(default_factory=dict)

    def reach(self) -> float:
        """Outer radius of the strained region (footprint + tent) [m]."""
        amp = 1.0 + sum(abs(a) + abs(b) for a, b in self.harmonics.values())
        return self.half_side * amp + self.tent_width

    def signed_distance(self, grid: Grid) -> np.ndarray:
        """Approximate signed distance to the footprint boundary (>0 outside) [m]."""
        X, Y = grid.mesh()
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        xr = c * X + s * Y
        yr = -s * X + c * Y
        theta = np.arctan2(yr, xr)
        rho = np.ones_like(theta)
        for k, (ak, bk) in self.harmonics.items():
            rho = rho + ak * np.cos(k * theta) + bk * np.sin(k * theta)
        rho = np.clip(rho, 0.2, 3.0)
        p = self.power
        r_se = (np.abs(xr / self.half_side) ** p + np.abs(yr / self.half_side) ** p) ** (1.0 / p)
        return (r_se - rho) * self.half_side

    def coverage(self, grid: Grid) -> np.ndarray:
        """Anti-aliased footprint: fraction of each pixel covered by the pillar top."""
        sd = self.signed_distance(grid)
        return np.clip(0.5 - sd / grid.dx, 0.0, 1.0)

    def footprint(self, grid: Grid) -> np.ndarray:
        return self.signed_distance(grid) <= 0.0


def periodic_distance(mask: np.ndarray, dx: float) -> np.ndarray:
    """Euclidean distance (m) from each pixel to the nearest True pixel, periodic."""
    tiled = np.tile(mask, (3, 3))
    d = distance_transform_edt(~tiled) * dx
    n = mask.shape[0]
    return d[n:2 * n, n:2 * n]


def _laplace_dirichlet(free: np.ndarray, known: np.ndarray, phi: np.ndarray | None = None,
                       theta_min: float = 0.1) -> np.ndarray:
    """Solve  Delta h = 0  on ``free`` with h = ``known`` elsewhere (periodic 5-point).

    If a level function ``phi`` (>0 on free nodes, <0 on Dirichlet nodes, continuous)
    is given, the boundary is placed sub-pixel by the Shortley-Weller cut-cell rule:
    for a boundary neighbour at parameter theta = phi_0 / (phi_0 - phi_nb) the link
    (h_nb - h_0) is weighted by 1/theta.  This removes the staircase corners of a
    pixelated mask, which otherwise act as artificial strain singularities.
    """
    nfree = int(free.sum())
    if nfree == 0:
        return known.copy()
    idx = -np.ones(free.shape, dtype=np.int64)
    idx[free] = np.arange(nfree)
    rows, cols, vals = [], [], []
    rhs = np.zeros(nfree)
    diag = np.zeros(nfree)
    own = idx[free]
    phi0 = phi[free] if phi is not None else None
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        nb_idx = np.roll(idx, shift, axis=axis)[free]
        nb_known = np.roll(known, shift, axis=axis)[free]
        nb_free = np.roll(free, shift, axis=axis)[free]
        w = np.ones(nfree)
        if phi is not None:
            phin = np.roll(phi, shift, axis=axis)[free]
            with np.errstate(divide="ignore", invalid="ignore"):
                theta = np.where(nb_free, 1.0, phi0 / (phi0 - phin))
            theta = np.clip(np.nan_to_num(theta, nan=1.0), theta_min, 1.0)
            w = 1.0 / theta
        diag += w
        rows.append(own[nb_free]); cols.append(nb_idx[nb_free]); vals.append(-w[nb_free])
        rhs[~nb_free] += (w * nb_known)[~nb_free]
    rows.append(own); cols.append(own); vals.append(diag)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nfree, nfree))
    if nfree <= 300_000:
        sol = spla.spsolve(A.tocsc(), rhs)
    else:
        sol, info = spla.cg(A, rhs, rtol=1e-10, maxiter=20000)
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info})")
    h = known.copy()
    h[free] = sol
    return h


def drape_height(pillar: Pillar, grid: Grid, rounding: float = 0.0) -> np.ndarray:
    """Graphene height profile h(x, y) [m] over one periodic cell."""
    sd = pillar.signed_distance(grid)
    phi = np.minimum(sd, pillar.tent_width - sd)     # >0 in the tent, <0 on pillar and floor
    free = phi > 0.0
    known = np.where(sd <= 0.0, pillar.height, 0.0)
    h = _laplace_dirichlet(free, known, phi=phi)
    if rounding > 0:
        h = gaussian_filter(h, sigma=rounding / grid.dx, mode="wrap")
    return h


def tent_mask(pillar: Pillar, grid: Grid, margin: float = 0.0) -> np.ndarray:
    """Region between the footprint and the adhered floor (where strain lives)."""
    sd = pillar.signed_distance(grid)
    return (sd > -margin) & (sd <= pillar.tent_width + margin)
