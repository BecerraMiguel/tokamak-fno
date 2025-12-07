"""
src/data/__init__.py

Módulo de datos para el proyecto Tokamak FNO.
Contiene generadores de datos sintéticos y DataLoaders de PyTorch.
"""

from .synthetic import SyntheticTokamakData
from .loader import TokamakDataset, get_dataloaders, print_dataset_info

__all__ = [
    'SyntheticTokamakGenerator',
    'TokamakDataset', 
    'get_dataloaders',
    'print_dataset_info'
]