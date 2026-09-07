"""Mesh calibration: how peak / rms pseudo-magnetic field depends on grid spacing
and on the physical rounding length of the pillar edge.

The pseudo-magnetic field is a derivative of strain, and strain is a derivative
of the height profile.  At a pillar edge of vanishing rounding the field is a
line singularity, so any quoted 'peak B_ps' is a statement about resolution.
The sweep below shows (i) that B_max grows without bound as dx -> 0 when
rounding = 0 and (ii) converges once dx is a few times smaller than the
rounding length, while integrated quantities (rms B in the tent, Raman-convolved
strain) converge already on coarse meshes.
"""
from __future__ import annotations

import time

import numpy as np

from .forward import run_forward, summarize
from .topography import Grid, Pillar


def fit_tent_width(pillar: Pillar, L: float, target_eps: float, N: int = 256,
                   rounding: float = 4e-9, relax: str = "pinned", lo: float = 60e-9,
                   hi: float | None = None, tol: float = 1e-9) -> Pillar:
    """Choose the tent width so that the peak atomic strain equals the value
    inferred from Raman (e.g. 2.4 % for the 500 nm / 85 nm pillars of Lu 2023).
    Strain decreases monotonically with tent width, so bisection suffices."""
    from dataclasses import replace
    if hi is None:
        hi = 0.5 * L - (pillar.reach() - pillar.tent_width) - 4 * L / N
    grid = Grid(L, N)

    def eps_max(w):
        res = run_forward(replace(pillar, tent_width=w), grid, rounding=rounding, relax=relax)
        return float(res.principal_strain.max())

    if eps_max(hi) > target_eps:
        return replace(pillar, tent_width=hi)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if eps_max(mid) > target_eps:
            lo = mid
        else:
            hi = mid
    return replace(pillar, tent_width=0.5 * (lo + hi))


def fit_rounding(pillar: Pillar, L: float, target_eps: float, N: int = 376,
                 relax: str = "pinned", lo: float = 4e-9, hi: float = 150e-9, tol: float = 0.5e-9) -> float:
    """Choose the edge-rounding length so that the peak atomic strain equals the
    Raman/tight-binding value reported for the fabricated pillars.  Wet (BOE) etching
    gives sloped, rounded pillar edges, so the rounding is a genuine sample parameter
    rather than a numerical device.  Peak strain decreases monotonically with rounding."""
    grid = Grid(L, N)

    def eps_max(r):
        res = run_forward(pillar, grid, rounding=r, relax=relax)
        return float(res.principal_strain.max())

    if eps_max(hi) > target_eps:
        return hi
    if eps_max(lo) < target_eps:
        return lo
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if eps_max(mid) > target_eps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mesh_sweep(pillar: Pillar, L: float, dx_list, rounding: float, relax: str = "pinned",
               B_thr: float = 20.0):
    rows = []
    for dx in dx_list:
        N = int(round(L / dx))
        N += N % 2
        grid = Grid(L, N)
        t = time.perf_counter()
        res = run_forward(pillar, grid, rounding=rounding, relax=relax)
        s = summarize(res, B_thr=B_thr)
        s["rounding_nm"] = rounding * 1e9
        s["time_s"] = time.perf_counter() - t
        rows.append(s)
    return rows


def convergence_table(rows, key: str):
    """Relative change of ``key`` between consecutive refinements."""
    vals = np.array([r[key] for r in rows])
    rel = np.abs(np.diff(vals)) / np.abs(vals[1:])
    return vals, rel


def print_rows(rows, keys=("dx_nm", "N", "eps_max_pct", "eps_raman_max_pct", "B_max_T",
                           "B_rms_tent_T", "coverage_area_um2", "equilibrium_residual", "time_s")):
    head = " | ".join(f"{k:>18s}" for k in keys)
    print(head)
    print("-" * len(head))
    for r in rows:
        print(" | ".join(f"{r[k]:>18.4g}" if isinstance(r[k], float) else f"{r[k]:>18}" for k in keys))
