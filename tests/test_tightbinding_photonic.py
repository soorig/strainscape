import numpy as np

from strainscape import photonic as ph
from strainscape import tightbinding as tb
from strainscape.constants import A0, BETA, HBAR_OVER_E


def test_pseudo_landau_levels_match_continuum_field():
    B = 200.0
    fl = tb.honeycomb_flake(7e-9)
    c = -B / (8 * HBAR_OVER_E * BETA / (2 * A0))
    H = tb.hamiltonian(fl, tb.triaxial_displacement(fl.pos, c))
    vals, vecs = tb.spectrum_near(H, k=80)
    rr = np.hypot(fl.pos[:, 0], fl.pos[:, 1])
    interior = rr < 4.5e-9
    bulk = np.sum(np.abs(vecs[interior]) ** 2, axis=0) > 0.6
    E1 = tb.landau_level_energies(B, 1)[-1]
    for sign in (+1, -1):
        near = vals[bulk][np.abs(vals[bulk] - sign * E1) < 0.1]
        assert len(near) >= 2
        assert abs(near.mean() - sign * E1) / E1 < 0.05
    wA, wB = tb.sublattice_weight(vecs, vals, fl.sub, 0.1, interior)
    assert max(wA, wB) > 10 * min(wA, wB)        # zeroth level is sublattice polarised


def test_pristine_flake_has_no_gap_structure_at_landau_energies():
    fl = tb.honeycomb_flake(7e-9)
    vals, _ = tb.spectrum_near(tb.hamiltonian(fl), k=40)
    assert np.all(np.abs(vals) < 0.5)            # dense states near the Dirac point


def test_coupling_coefficient_converges_with_mesh():
    k5 = ph.coupling_coefficient(200e-9, dx=5e-9)
    k25 = ph.coupling_coefficient(200e-9, dx=2.5e-9)
    assert abs(k5 - k25) / k25 < 0.01
    assert ph.coupling_coefficient(300e-9, dx=2.5e-9) < k25


def test_photonic_landau_levels():
    gaps = np.array([150, 200, 250, 300]) * 1e-9
    _, ell, kappa_at = ph.calibrate_coupling(gaps, [5e-9, 2.5e-9])
    d0 = 650e-9
    kappa0 = float(kappa_at(d0 - 450e-9))
    fl, H, beta_ph = ph.photonic_lattice(6.0, d0, ell, kappa0, c=0.0)
    Bph = 1.0 / (2 * d0) ** 2
    c = -Bph / (8 * beta_ph / (2 * d0))
    _, Hs, _ = ph.photonic_lattice(6.0, d0, ell, kappa0, c=c)
    vals, vecs = np.linalg.eigh(Hs)
    rr = np.hypot(fl.pos[:, 0], fl.pos[:, 1])
    bulk = np.sum(np.abs(vecs[rr < 3.5 * d0]) ** 2, axis=0) > 0.6
    E1 = ph.photonic_landau_levels(Bph, kappa0, d0, 1)[-1]
    near = vals[bulk][np.abs(vals[bulk] - E1) < 0.3 * E1]
    assert len(near) >= 1
    assert abs(near.mean() - E1) / E1 < 0.15
