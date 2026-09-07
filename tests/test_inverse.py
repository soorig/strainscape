import numpy as np

from strainscape import Grid, Pillar
from strainscape import inverse as inv


def test_surrogate_matches_pillar_geometry():
    grid = Grid(1.5e-6, 96)
    pil = Pillar(half_side=250e-9, power=8, height=85e-9, tent_width=400e-9, harmonics={4: (0.1, 0.0)})
    h = inv.surrogate_height(pil, grid, rounding=0.0)
    sd = pil.signed_distance(grid)
    assert np.allclose(h[sd <= 0], pil.height)
    assert np.allclose(h[sd >= pil.tent_width], 0.0)


def test_optimizer_runs_and_respects_bounds():
    space = inv.DesignSpace(L=1.5e-6)
    grid = Grid(1.5e-6, 96)
    x0 = (250e-9, 8.0, 400e-9, 0.0, 0.0)
    out = inv.optimize(space, x0, grid, rounding=15e-9, maxiter=3, verbose=False)
    assert np.all(out.x >= 0) and np.all(out.x <= 1)
    assert out.fun <= out.history[0][1] + 1e-12
    lo, hi = space.bounds["half_side"]
    assert lo <= out.pillar.half_side <= hi
