"""strainscape: strain -> pseudo-magnetic-field pipeline for graphene on nanopillars,
with mesh calibration, tight-binding validation, inverse design and a photonic twin."""

from .topography import Grid, Pillar, drape_height, tent_mask
from .elasticity import strain_from_height
from .pmf import (gauge_field, pseudo_magnetic_field, sublattice_polarization,
                  pmf_from_polarization, uniform_field_from_triaxial)
from .forward import run_forward, summarize, raman_convolve, ForwardResult

__all__ = [
    "Grid", "Pillar", "drape_height", "tent_mask", "strain_from_height",
    "gauge_field", "pseudo_magnetic_field", "sublattice_polarization",
    "pmf_from_polarization", "uniform_field_from_triaxial",
    "run_forward", "summarize", "raman_convolve", "ForwardResult",
]
__version__ = "0.1.0"
