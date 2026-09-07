# Model notes

## 1. Geometry and drape

One pillar per square periodic cell of pitch `L`. The footprint is a super-ellipse
`|x/s|^p + |y/s|^p = rho(theta)^p` with `rho = 1 + sum_k a_k cos(k theta) + b_k sin(k theta)`;
`p` sets the corner sharpness (a fabricated wet-etched pillar has soft corners, `p ~ 6-8`),
the harmonics allow star-like or lobed footprints that are still a single e-beam exposure.

The sheet is pinned at height `H` on the pillar top and adhered (`h = 0`) beyond a tent width
`w` from the footprint. In between it is a tension-dominated membrane, `Laplace h = 0`.
The Dirichlet boundaries are placed sub-pixel with the Shortley-Weller cut-cell stencil,
otherwise the pixel staircase of the mask acts as a set of artificial corners whose strain
singularities never converge under refinement (this was the first thing the mesh study caught).

The kink of the membrane at the pillar edge is smoothed by a Gaussian of radius `r`
("edge rounding"). For BOE-etched SiO2 pillars the sloped sidewall makes `r` a *sample*
parameter of order tens of nm; it is fitted so that the peak atomic strain equals the value
reported for the fabricated array (2.4 % in Lu 2023). The mesh is then converged when
`dx <= r/4`.

## 2. Strain

Small-slope Foeppl-von Karman kinematics with prescribed out-of-plane profile:

    eps_ij = 1/2 (d_i u_j + d_j u_i) + 1/2 d_i h d_j h

* `relax = "pinned"` (default): `u = 0`. The sheet adheres where it lands; the extra area
  needed by the tent is supplied by slack during transfer; the flat floor stays unstrained,
  as in the Raman maps of Kang 2021 / Lu 2023.
* `relax = "free"`: `u` minimises the plane-stress membrane energy inside the periodic cell,
  solved exactly by FFT with the central-difference symbol so that `div sigma = 0` holds on
  the grid to machine precision (`equilibrium_residual ~ 1e-15`). This is the frictionless
  lower bound on strain inhomogeneity (peak strain roughly 4x smaller, peak field ~8x smaller).

Graphene constants: `E_2D = 340 N/m`, `nu = 0.165`.

## 3. Gauge field, pseudo-magnetic field, sublattice polarisation

With `x` along the zigzag direction and `A -> B` bond vectors
`delta_1 = a0(sqrt3/2, 1/2)`, `delta_2 = a0(-sqrt3/2, 1/2)`, `delta_3 = a0(0, -1)`:

    A    = (hbar/e) (beta / 2 a0) (eps_xx - eps_yy, -2 eps_xy)         [T m]
    B_ps = d_x A_y - d_y A_x                                            [T]
    P    = sum_n (d_n / a0) delta_n = (3 a0/4) (2 eps_xy, eps_xx - eps_yy)   [m]
    B_ps = -(hbar/e) (2 beta / 3 a0^2) div P

The last line is Eq. (3) of Lu et al. 2023. Derivation: `sum_n n_i n_j n_k` for the three
bond directions has the only non-zero components `sum n_x n_x n_y = 3/4`, `sum n_y^3 = -3/4`,
which gives `P = (3 a0/4)(2 eps_xy, eps_xx - eps_yy) = (3 a0/2 beta) (A_y, -A_x) (e/hbar)`, hence
`div P = -(3 a0/2 beta)(e/hbar) B_ps`. `tests/test_pmf.py` checks the identity to `1e-15`.

Scale: a 3 % change of `eps_xx - eps_yy` across 5 nm gives `B_ps ~ 40 T`.

`beta = 3`, `a0 = 0.142 nm`, `t0 = 2.7 eV`, `v_F = 3 t0 a0 / 2 hbar = 0.87e6 m/s`.

## 4. Tight-binding validation

`H_ij = -t0 exp(-beta (d_ij / a0 - 1))` on the honeycomb graph, with `d_ij` from the displaced
positions. A disc flake with the triaxial displacement `u = c (2xy, x^2 - y^2)` has
`eps_xx - eps_yy = 4cy`, `eps_xy = 2cx`, so the field is uniform, `B_ps = -8c (hbar/e) beta / 2a0`.
Bulk eigenvalues (weight > 0.6 inside 65 % of the radius) reproduce
`E_n = sgn(n) v_F sqrt(2 e hbar |B| |n|)` for `n = +-1, +-2` within 1 %, and the `n = 0`
level has > 99 % of its weight on one sublattice (the mechanism of the strain-enabled SHG).

Zero modes of a bipartite graph live on the majority sublattice with multiplicity at least
`|n_A - n_B|` (rank argument); the flake edges therefore carry exact zero modes, which is why
the shift-invert solver is run at `sigma = 1.3e-4 eV` rather than `0`.

## 5. Mesh calibration

For each rounding `r` in `{r_cal, r_cal/2, r_cal/4, 0}` the forward model is run at
`dx = 16, 8, 4, 2 nm`. Peak `|B_ps|` and peak strain converge once `dx <= r/4`; with `r = 0`
the peak field grows roughly like `dx^(-1.3)` and never converges. The Raman-convolved strain
(361 nm Gaussian) and the rms field over the tent converge on the coarsest mesh. Conclusion:
a quoted peak `B_ps` of a continuum model is a statement about `r`; compare models on
integrated quantities or quote `r` alongside.

## 6. Inverse design

Design variables: `s, p, w, a_4, a_8` (height fixed by the etch). With the cut-cell boundary
the Laplace drape is continuous in these variables, so the default optimiser is Powell
(derivative-free) on the true drape at `dx = 8 nm`, followed by re-evaluation at `dx = 2 nm`.
A smooth logarithmic surrogate tent (the exact drape of a disc pillar evaluated on the
super-elliptic distance) is available for L-BFGS-B screening; the earlier linear surrogate
under-predicted the peak strain by 2x and produced optima that violated the strain cap once
verified, which is why the optimiser now runs on the true drape.
Objective: maximise the cell-averaged `|B_ps|` subject to peak strain `<= 3 %` (fracture /
slip margin) and the strained region staying inside the cell. Alternatives in
`inverse.py`: area with `|B| >= B_thr`, and the coefficient of variation of `|B|`
(uniform-field target, the limitation named in the discussion of Kang 2021).

Result for a 10 nm edge: the 500 nm square sits at 6 % peak strain; the optimiser meets the
3 % cap at an unchanged mean field (0.5 T) by rounding the footprint (`p -> 2`) and widening
the tent. No footprint shape raises the mean field at fixed pitch and height: the mean field
is edge sharpness times edge length per unit area, so pitch and pillar count per cell are the
design variables to add next (the `Pillar` / `Grid` split makes that a small change).

The adjoint gradient is straightforward for this chain (linear Laplace solve, quadratic
strain, linear field) and is the natural next step once the design space grows beyond a
handful of parameters.

## 7. Photonic twin

A honeycomb array of coupled waveguides obeys `i dpsi/dz = H psi` with `H_ij = kappa(d_ij)`.
The coupling law is *calibrated* from a 1D finite-difference mode solver of two parallel Si
slabs (450 nm wide, `n = 3.48 / 1.444`, 1550 nm): `kappa = (beta_even - beta_odd)/2`, converged
to < 1 % at `dx = 10 nm`; the fit `kappa = kappa0 exp(-(d - d0)/ell)` gives `ell ~ 85 nm`, hence
`beta_ph = d0/ell ~ 7.6` at `d0 = 650 nm`. The same triaxial displacement produces photonic
pseudo-Landau levels at `E_n = sgn(n) (3 kappa0 d0 / 2) sqrt(2 |B| |n|)`, and the walk
`exp(-iHz)` launched at the central waveguide stays put (flat bands), the photonic analogue
of the slow carrier dynamics of Kang 2021.

## 8. Limitations

* Continuum small-slope membrane; no bending stiffness, no adhesion mechanics, no
  self-consistent tent width. The tent width and edge rounding are calibrated, not predicted.
* `beta = 3` and the exponential hopping law are the standard first-order model; the papers
  use the same.
* Peak fields are resolution-limited by construction (see section 5); the atomistic peak of
  ~30 T (Lu 2023) or ~100 T (Kang 2021) corresponds to `r` at the nanometre scale.
* The photonic twin uses a scalar 1D slab coupling; a 2D vector mode solver or an FDTD run
  (Tidy3D / Meep) would replace `calibrate_coupling` without changing anything downstream.
