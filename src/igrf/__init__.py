"""
use IGRF via f2py from Python
"""

__version__ = "13.0.2"

from .base import igrf, grid, build
from .utils import mag_vector2incl_decl, latlon2colat, latlonworldgrid
