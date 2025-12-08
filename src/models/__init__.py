"""
src/models/__init__.py

Módulo de modelos para predicción de disrupciones en tokamaks.
Exporta los modelos y utilidades principales.
"""

from .baseline import BaselineCNN, count_parameters

__all__ = ['BaselineCNN', 'count_parameters']