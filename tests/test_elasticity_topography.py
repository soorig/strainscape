import numpy as np

from strainscape import Grid, Pillar, drape_height, run_forward
from strainscape.elasticity import equilibrium_residual, strain_from_height


def test_drape_is_bounded_and_pinned():
    grid = Grid(1.5e-6, 128)
    pil = Pillar(half_side=250e-9, power=8, height=85e-9, tent_width=400e-9)
    h = drape_height(pil, grid)
    assert h.min() >= -1e-12 and h.max() <= pil.height + 1e-12
    sd = pil.signed_distance(grid)
    assert np.allclose(h[sd <= 0], pil.height)
    assert np.allclose(h[sd > pil.tent_width], 0.0)
    row = h[grid.N // 2, grid.N // 2:]
    assert np.all(np.diff(row) <= 1e-12)          # monotone decrease away from the pillar


def test_free_relaxation_is_in_equilibrium():
    grid = Grid(1.5e-6, 128)
    pil = Pillar(half_side=250e-9, power=6, height=85e-9, tent_width=400e-9)
    h = drape_height(pil, grid, rounding=20e-9)
    exx, eyy, exy, ux, uy = strain_from_height(h, grid.dx, relax="free")
    assert equilibrium_residual(exx, eyy, exy, grid.dx) < 1e-8
    # relaxation lowers the strain energy density peak relative to the pinned sheet
    gxx, gyy, gxy, _, _ = strain_from_height(h, grid.dx, relax="pinned")
    assert (exx ** 2 + eyy ** 2 + 2 * exy ** 2).max() < (gxx ** 2 + gyy ** 2 + 2 * gxy ** 2).max()


def test_peak_field_converges_with_rounding():
    pil = Pillar(half_side=250e-9, power=8, height=85e-9, tent_width=488e-9)
    r = 24e-9
    b = [np.abs(run_forward(pil, Grid(1.5e-6, N), rounding=r).B).max() for N in (126, 250)]
    assert abs(b[1] - b[0]) / b[1] < 0.1


def test_peak_field_grows_without_rounding():
    pil = Pillar(half_side=250e-9, power=8, height=85e-9, tent_width=488e-9)
    b = [np.abs(run_forward(pil, Grid(1.5e-6, N), rounding=0.0).B).max() for N in (126, 250)]
    assert b[1] > 1.5 * b[0]
