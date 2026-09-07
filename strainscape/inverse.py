"""Inverse design of a fabricable pillar footprint for a target B_ps landscape.

Design variables (all realisable with e-beam lithography of the pillar and a
fixed etch depth):  half side, super-ellipse exponent (corner sharpness),
tent width (set by adhesion / transfer), and 4-fold / 8-fold radial harmonics
of the footprint.  Height is fixed (85 nm in Lu et al. 2023).

For gradient-based optimisation the drape is replaced by a smooth surrogate
h = H * S(signed distance / tent width) that is differentiable in the design
variables (the Laplace drape is piecewise constant in them on a pixel grid).
The optimum is then re-evaluated with the full Laplace drape on a fine mesh.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

from .forward import ForwardResult, run_forward, summarize
from .topography import Grid, Pillar


@dataclass
class DesignSpace:
    L: float = 1500e-9
    height: float = 85e-9
    bounds: dict = field(default_factory=lambda: {
        "half_side": (120e-9, 400e-9),
        "power": (2.0, 16.0),
        "tent_width": (100e-9, 550e-9),
        "a4": (-0.25, 0.25),
        "a8": (-0.12, 0.12),
    })
    names: tuple = ("half_side", "power", "tent_width", "a4", "a8")

    def to_pillar(self, x: np.ndarray) -> Pillar:
        hs, p, w, a4, a8 = self.unscale(x)
        return Pillar(half_side=hs, power=p, height=self.height, tent_width=w,
                      harmonics={4: (a4, 0.0), 8: (a8, 0.0)})

    # optimiser works on O(1) variables in [0, 1]
    def scale(self, values) -> np.ndarray:
        return np.array([(v - lo) / (hi - lo) for v, (lo, hi) in zip(values, self.bounds.values())])

    def unscale(self, x) -> list:
        return [lo + xi * (hi - lo) for xi, (lo, hi) in zip(x, self.bounds.values())]


def surrogate_height(pillar: Pillar, grid: Grid, rounding: float) -> np.ndarray:
    """Smooth logarithmic tent: the exact Laplace drape of a *disc* pillar of radius R = half_side
    with the floor at R + w, evaluated on the signed super-elliptic distance,
        h = H ln((R + w) / (R + sd)) / ln((R + w) / R),   0 <= sd <= w.
    Differentiable in every design variable; slope is largest at the pillar edge like the
    true drape (the linear tent used earlier under-predicted the peak strain by ~2x)."""
    sd = np.clip(pillar.signed_distance(grid), 0.0, pillar.tent_width)
    R, w = pillar.half_side, pillar.tent_width
    h = pillar.height * np.log((R + w) / (R + sd)) / np.log((R + w) / R)
    if rounding > 0:
        h = gaussian_filter(h, sigma=rounding / grid.dx, mode="wrap")
    return h


def evaluate(pillar: Pillar, grid: Grid, rounding: float, surrogate: bool = True,
             relax: str = "pinned") -> ForwardResult:
    h = surrogate_height(pillar, grid, rounding) if surrogate else None
    return run_forward(pillar, grid, rounding=rounding, relax=relax, h=h)


def _penalties(res: ForwardResult, eps_cap: float, L: float) -> float:
    """Strain cap (fracture / slip margin) and 'strained region stays inside the cell'."""
    eps_max = float(res.principal_strain.max())
    pen = 1e4 * max(0.0, eps_max - eps_cap) ** 2 / eps_cap ** 2
    pen += 200.0 * max(0.0, res.pillar.reach() / (0.47 * L) - 1.0) ** 2
    return pen


def objective_mean_field(res: ForwardResult, eps_cap: float, L: float, B_scale: float = 10.0, **_) -> float:
    """-<|B|>_cell / B_scale + penalties.  The cell-averaged |B| is the mesh-robust,
    pump-probe-relevant quantity (Kang 2021 fit an *average* B_ps ~ 25 T)."""
    return -float(np.mean(np.abs(res.B))) / B_scale + _penalties(res, eps_cap, L)


def objective_coverage(res: ForwardResult, B_thr: float, eps_cap: float, L: float, **_) -> float:
    """-(area fraction with |B| >= B_thr) + penalties."""
    cover = float(np.mean(np.abs(res.B) >= B_thr))
    return -cover + _penalties(res, eps_cap, L)


def objective_uniformity(res: ForwardResult, eps_cap: float, L: float, B_floor_frac: float = 0.1, **_) -> float:
    """Coefficient of variation of |B| where |B| > frac * max (uniform-field target)."""
    absB = np.abs(res.B)
    m = absB > B_floor_frac * absB.max()
    return float(absB[m].std() / absB[m].mean()) + _penalties(res, eps_cap, L)


@dataclass
class OptResult:
    x: np.ndarray
    pillar: Pillar
    history: list
    initial: Pillar
    fun: float


def optimize(space: DesignSpace, x0_values, grid: Grid, rounding: float,
             objective=objective_mean_field, B_thr: float = 20.0, eps_cap: float = 0.03,
             maxiter: int = 60, verbose: bool = True, surrogate: bool = False,
             method: str = "Powell") -> OptResult:
    """Optimise the footprint.

    surrogate=False (default): objective evaluated on the true Laplace drape.  With the
    cut-cell boundary the drape is continuous in the design variables, and the
    derivative-free Powell method copes with the residual pixel-level roughness.
    surrogate=True: smooth logarithmic tent, allows L-BFGS-B with FD gradients (fast screening).
    """
    history = []

    def f(x):
        x = np.clip(x, 0.0, 1.0)
        pil = space.to_pillar(x)
        res = evaluate(pil, grid, rounding, surrogate=surrogate)
        val = objective(res, B_thr=B_thr, eps_cap=eps_cap, L=space.L)
        history.append((x.copy(), val))
        if verbose and len(history) % 25 == 0:
            print(f"  eval {len(history):4d}  f = {val:+.5f}")
        return val

    x0 = space.scale(x0_values)
    bounds = [(0.0, 1.0)] * len(x0)
    if method == "Powell":
        out = minimize(f, x0, method="Powell", bounds=bounds,
                       options={"maxiter": maxiter, "maxfev": 40 * maxiter, "xtol": 1e-3, "ftol": 1e-6})
    else:
        out = minimize(f, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "eps": 2e-3, "ftol": 1e-9})
    best = min(history, key=lambda t: t[1])
    return OptResult(best[0], space.to_pillar(best[0]), history, space.to_pillar(x0), best[1])


def verify(pillar: Pillar, L: float, N: int, rounding: float, B_thr: float = 20.0) -> dict:
    """Re-evaluate a design with the full Laplace drape on an N x N mesh."""
    res = run_forward(pillar, Grid(L, N), rounding=rounding)
    return summarize(res, B_thr=B_thr), res
