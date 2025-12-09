"""
src/training/__init__.py

Módulo de entrenamiento para predicción de disrupciones.
"""

from .train import train_epoch, evaluate, train_model, EarlyStopping
from .evaluate import (
    evaluate_model,
    get_predictions,
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    print_evaluation_report
)

__all__ = ['train_epoch', 
           'evaluate', 
           'train_model', 
           'EarlyStopping',
           'evaluate_model',
            'get_predictions',
            'calculate_metrics',
            'plot_confusion_matrix',
            'plot_roc_curve',
            'plot_precision_recall_curve',
            'print_evaluation_report']