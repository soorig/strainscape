import numpy as np

from strainscape.constants import A0, BETA, HBAR_OVER_E
from strainscape.elasticity import ddx, ddy
from strainscape.pmf import (pmf_from_polarization, pseudo_magnetic_field,
                             sublattice_polarization, uniform_field_from_triaxial)


def _random_smooth(N, rng):
    k = np.fft.fftfreq(N)
    KX, KY = np.meshgrid(k, k)
    f = rng.standard_normal((N, N)) * np.exp(-((KX ** 2 + KY ** 2) * 200))
    return np.real(np.fft.ifft2(np.fft.fft2(f)))


def test_curl_A_equals_div_P_identity():
    rng = np.random.default_rng(0)
    N, dx = 64, 2e-9
    exx, eyy, exy = (0.01 * _random_smooth(N, rng) for _ in range(3))
    B1 = pseudo_magnetic_field(exx, eyy, exy, dx)
    Px, Py = sublattice_polarization(exx, eyy, exy)
    B2 = pmf_from_polarization(Px, Py, dx)
    assert np.max(np.abs(B1 - B2)) < 1e-9 * np.max(np.abs(B1))


def test_triaxial_displacement_gives_uniform_field():
    N, dx = 64, 1e-9
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    c = 1e6
    ux, uy = 2 * c * X * Y, c * (X ** 2 - Y ** 2)
    exx, eyy = ddx(ux, dx), ddy(uy, dx)
    exy = 0.5 * (ddy(ux, dx) + ddx(uy, dx))
    B = pseudo_magnetic_field(exx, eyy, exy, dx)
    inner = (slice(4, -4), slice(4, -4))       # central differences are exact for quadratics away from the wrap
    assert np.allclose(B[inner], uniform_field_from_triaxial(c), rtol=1e-8)


def test_field_scale_is_tesla_like():
    # strain difference 3 % changing over 5 nm  ->  tens of tesla (Kang 2021 / Lu 2023 scale)
    pref = HBAR_OVER_E * BETA / (2 * A0)
    assert 20 < pref * 0.03 / 5e-9 < 60
