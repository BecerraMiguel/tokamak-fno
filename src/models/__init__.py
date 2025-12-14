"""
src/models/__init__.py

Módulo de modelos para predicción de disrupciones en tokamaks.
Exporta los modelos y utilidades principales.
"""

from .baseline import BaselineCNN, count_parameters
from .fno import SpectralConv1d, FourierLayer, FNO1d, count_fno_parameters

__all__ = ['BaselineCNN', 
           'count_parameters',
           'SpectralConv1d',
            'FourierLayer', 
            'FNO1d',
            'count_fno_parameters']