"""Physical constants and graphene parameters.

All lengths in metres, energies in eV unless stated otherwise.
Values follow the conventions used in Kang et al., Nat. Commun. 12, 5087 (2021)
and Lu et al., Nat. Commun. 14, 2580 (2023):
    A = (beta / 2 a0) * (eps_xx - eps_yy, -2 eps_xy)      (gauge field, x along zigzag)
    B_ps = (2 beta / 3 a0^2) * div P                       (from sublattice polarization)
"""

A0 = 0.142e-9            # C-C bond length [m]
BETA = 3.0               # -d ln t / d ln a  (papers use ~3)
T0 = 2.7                 # nearest-neighbour hopping [eV]
HBAR_OVER_E = 6.582119569e-16   # hbar / e  [V s] = [T m^2]
HBAR = 1.054571817e-34   # [J s]
E_CHARGE = 1.602176634e-19       # [C]

# graphene 2D elastic constants (Lee et al., Science 321, 385 (2008))
E2D = 340.0              # 2D Young's modulus [N/m]
NU = 0.165               # Poisson ratio

# derived Lame parameters for plane stress in 2D
MU_2D = E2D / (2.0 * (1.0 + NU))            # shear modulus [N/m]
LAMBDA_2D = E2D * NU / (1.0 - NU ** 2)      # 2D Lame lambda [N/m]

# Fermi velocity for the nearest-neighbour model: v_F = 3 t0 a0 / (2 hbar)
V_F = 3.0 * T0 * E_CHARGE * A0 / (2.0 * HBAR)   # [m/s]  (~0.87e6)

# Raman spot (532 nm, NA 0.9): 0.61 lambda / NA = 361 nm  (Kang 2021, Eq. 2)
RAMAN_FWHM = 0.61 * 532e-9 / 0.9
