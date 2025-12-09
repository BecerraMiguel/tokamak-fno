"""
Módulo de evaluación detallada para predicción de disrupciones.

Este módulo implementa métricas específicas para el dominio de fusión nuclear,
donde la detección de disrupciones (True Positive Rate alto) es crítica
mientras se minimizan falsas alarmas (False Positive Rate bajo).

Métricas implementadas:
    - Confusion Matrix
    - TPR (True Positive Rate / Recall / Sensitivity)
    - FPR (False Positive Rate)
    - Precision, F1 Score
    - ROC Curve y AUC
    - Precision-Recall Curve
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)
from typing import Dict, Tuple, Optional
from pathlib import Path


def get_predictions(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Obtiene predicciones del modelo para todo un DataLoader.
    
    Esta función ejecuta el modelo en modo evaluación sobre todos los
    batches del DataLoader y recolecta las etiquetas reales, las
    predicciones de clase, y las probabilidades (para curvas ROC).
    
    Args:
        model: Modelo PyTorch entrenado
        data_loader: DataLoader con datos a evaluar
        device: Dispositivo (cuda/cpu)
        
    Returns:
        Tuple con tres arrays numpy:
            - y_true: Etiquetas reales [n_samples]
            - y_pred: Predicciones de clase (0 o 1) [n_samples]
            - y_proba: Probabilidades de clase positiva [n_samples]
    """
    model.eval()
    model.to(device)
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            
            # Probabilidades con softmax
            probs = torch.softmax(outputs, dim=1)
            
            # Predicciones (clase con mayor probabilidad)
            _, preds = torch.max(outputs, dim=1)
            
            # Guardar resultados
            all_labels.extend(batch_y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            # Probabilidad de clase positiva (disrupción = clase 1)
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calcula todas las métricas de clasificación relevantes.
    
    Para predicción de disrupciones, las métricas más importantes son:
    - TPR (Recall): Queremos detectar TODAS las disrupciones
    - FPR: Queremos minimizar falsas alarmas
    
    Args:
        y_true: Etiquetas reales
        y_pred: Predicciones de clase
        y_proba: Probabilidades de clase positiva
        
    Returns:
        Diccionario con todas las métricas calculadas
    """
    # Matriz de confusión
    # [[TN, FP], [FN, TP]] cuando labels=[0,1]
    cm = confusion_matrix(y_true, y_pred)
    
    # Extraer valores de la matriz
    # Para clasificación binaria con clases 0 (normal) y 1 (disruptivo)
    tn, fp, fn, tp = cm.ravel()
    
    # Calcular métricas
    # TPR = TP / (TP + FN) - Proporción de disrupciones detectadas
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # FPR = FP / (FP + TN) - Proporción de falsas alarmas
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # TNR (Specificity) = TN / (TN + FP)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # F1 Score = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
    
    # Accuracy = (TP + TN) / Total
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # AUC-ROC
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr_curve, tpr_curve)
    
    # Average Precision (área bajo curva PR)
    avg_precision = average_precision_score(y_true, y_proba)
    
    return {
        'accuracy': accuracy,
        'tpr': tpr,           # Recall / Sensitivity
        'fpr': fpr,
        'tnr': tnr,           # Specificity  
        'precision': precision,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total_samples': len(y_true),
        'total_disruptive': int(tp + fn),
        'total_normal': int(tn + fp)
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Genera visualización de matriz de confusión.
    
    La matriz muestra:
    - Cuadrante superior izquierdo: TN (Normal correctamente clasificado)
    - Cuadrante superior derecho: FP (Falsa alarma)
    - Cuadrante inferior izquierdo: FN (Disrupción no detectada - CRÍTICO)
    - Cuadrante inferior derecho: TP (Disrupción detectada correctamente)
    
    Args:
        y_true: Etiquetas reales
        y_pred: Predicciones
        save_path: Ruta para guardar la figura (opcional)
        figsize: Tamaño de la figura
        
    Returns:
        Objeto Figure de matplotlib
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Crear heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Etiquetas
    classes = ['Normal', 'Disruptivo']
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='Etiqueta Real',
        xlabel='Predicción del Modelo',
        title='Matriz de Confusión\nPredicción de Disrupciones en Tokamak'
    )
    
    # Rotar etiquetas del eje x
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Añadir texto en cada celda
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Determinar nombre de la métrica
            if i == 0 and j == 0:
                label = f'TN\n{cm[i, j]}'
            elif i == 0 and j == 1:
                label = f'FP\n{cm[i, j]}'
            elif i == 1 and j == 0:
                label = f'FN\n{cm[i, j]}'
            else:
                label = f'TP\n{cm[i, j]}'
                
            ax.text(j, i, label,
                   ha='center', va='center', fontsize=14,
                   color='white' if cm[i, j] > thresh else 'black')
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Matriz de confusión guardada en: {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Genera curva ROC (Receiver Operating Characteristic).
    
    La curva ROC muestra el trade-off entre TPR y FPR a diferentes
    umbrales de clasificación. El área bajo la curva (AUC) indica
    la capacidad discriminativa del modelo:
    - AUC = 1.0: Clasificador perfecto
    - AUC = 0.5: Clasificador aleatorio
    
    Args:
        y_true: Etiquetas reales
        y_proba: Probabilidades de clase positiva
        save_path: Ruta para guardar (opcional)
        figsize: Tamaño de figura
        
    Returns:
        Objeto Figure de matplotlib
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Curva ROC
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'Curva ROC (AUC = {roc_auc:.3f})')
    
    # Línea diagonal (clasificador aleatorio)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Aleatorio (AUC = 0.5)')
    
    # Punto de operación ideal (esquina superior izquierda)
    ax.scatter([0], [1], s=100, c='green', marker='*', zorder=5,
              label='Punto ideal (0, 1)')
    
    # Marcar targets del proyecto
    ax.axhline(y=0.90, color='red', linestyle=':', alpha=0.7,
               label='Target TPR > 90%')
    ax.axvline(x=0.10, color='red', linestyle=':', alpha=0.7,
               label='Target FPR < 10%')
    
    # Región de operación aceptable
    ax.fill_between([0, 0.10], [0.90, 0.90], [1, 1], alpha=0.1, color='green',
                    label='Zona objetivo')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Tasa de Falsas Alarmas)')
    ax.set_ylabel('True Positive Rate (Tasa de Detección)')
    ax.set_title('Curva ROC - Predicción de Disrupciones')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Curva ROC guardada en: {save_path}")
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Genera curva Precision-Recall.
    
    Esta curva es especialmente útil cuando las clases están desbalanceadas.
    Muestra el trade-off entre precision y recall a diferentes umbrales.
    
    Args:
        y_true: Etiquetas reales
        y_proba: Probabilidades de clase positiva
        save_path: Ruta para guardar (opcional)
        figsize: Tamaño de figura
        
    Returns:
        Objeto Figure de matplotlib
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Curva PR
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'Curva PR (AP = {avg_precision:.3f})')
    
    # Línea base (proporción de positivos)
    baseline = y_true.sum() / len(y_true)
    ax.axhline(y=baseline, color='navy', linestyle='--',
               label=f'Baseline ({baseline:.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (True Positive Rate)')
    ax.set_ylabel('Precision')
    ax.set_title('Curva Precision-Recall - Predicción de Disrupciones')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Curva PR guardada en: {save_path}")
    
    return fig


def print_evaluation_report(metrics: Dict[str, float]) -> None:
    """
    Imprime un reporte formateado de todas las métricas.
    
    Incluye indicadores visuales de si cada métrica alcanza
    los targets del proyecto.
    
    Args:
        metrics: Diccionario de métricas calculadas
    """
    print("\n" + "=" * 70)
    print("           REPORTE DE EVALUACIÓN - PREDICCIÓN DE DISRUPCIONES")
    print("=" * 70)
    
    # Resumen de datos
    print(f"\n📊 RESUMEN DEL DATASET")
    print(f"   Total de muestras:     {metrics['total_samples']}")
    print(f"   Disparos disruptivos:  {metrics['total_disruptive']}")
    print(f"   Disparos normales:     {metrics['total_normal']}")
    
    # Matriz de confusión resumida
    print(f"\n📋 MATRIZ DE CONFUSIÓN")
    print(f"   ┌─────────────────────────────────────┐")
    print(f"   │  TN: {metrics['tn']:4d}  │  FP: {metrics['fp']:4d}  │")
    print(f"   │  FN: {metrics['fn']:4d}  │  TP: {metrics['tp']:4d}  │")
    print(f"   └─────────────────────────────────────┘")
    
    # Métricas principales con targets
    print(f"\n🎯 MÉTRICAS PRINCIPALES")
    
    # TPR
    tpr_status = "✅" if metrics['tpr'] >= 0.90 else "⚠️"
    print(f"   {tpr_status} TPR (Recall):     {metrics['tpr']:.4f} ({metrics['tpr']*100:.1f}%)  [Target: >90%]")
    
    # FPR
    fpr_status = "✅" if metrics['fpr'] <= 0.10 else "⚠️"
    print(f"   {fpr_status} FPR:              {metrics['fpr']:.4f} ({metrics['fpr']*100:.1f}%)  [Target: <10%]")
    
    # Otras métricas
    print(f"\n📈 MÉTRICAS ADICIONALES")
    print(f"   Accuracy:         {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"   Precision:        {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
    print(f"   F1 Score:         {metrics['f1']:.4f}")
    print(f"   Specificity:      {metrics['tnr']:.4f} ({metrics['tnr']*100:.1f}%)")
    print(f"   ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"   Avg Precision:    {metrics['avg_precision']:.4f}")
    
    # Interpretación
    print(f"\n💡 INTERPRETACIÓN")
    if metrics['tpr'] >= 0.90 and metrics['fpr'] <= 0.10:
        print("   ✅ El modelo cumple con los targets del proyecto.")
        print("   ✅ Listo para siguiente fase (implementación FNO).")
    elif metrics['tpr'] >= 0.90:
        print("   ✅ Detección de disrupciones excelente.")
        print("   ⚠️ Tasa de falsas alarmas por encima del target.")
    elif metrics['fpr'] <= 0.10:
        print("   ⚠️ Algunas disrupciones no detectadas (revisar FN).")
        print("   ✅ Pocas falsas alarmas.")
    else:
        print("   ⚠️ Modelo necesita mejoras en ambas métricas.")
    
    print("\n" + "=" * 70)


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: Optional[str] = None,
    plot: bool = True
) -> Dict[str, float]:
    """
    Función principal de evaluación completa.
    
    Ejecuta el pipeline completo de evaluación:
    1. Obtiene predicciones del modelo
    2. Calcula todas las métricas
    3. Genera visualizaciones (opcional)
    4. Imprime reporte formateado
    
    Args:
        model: Modelo PyTorch entrenado
        data_loader: DataLoader con datos de evaluación
        device: Dispositivo (cuda/cpu)
        save_dir: Directorio para guardar plots (opcional)
        plot: Si generar visualizaciones
        
    Returns:
        Diccionario con todas las métricas
        
    Example:
        >>> model = BaselineCNN(in_channels=5, num_classes=2)
        >>> model.load_state_dict(torch.load('results/best_model.pt'))
        >>> metrics = evaluate_model(model, val_loader, device, 'results/')
    """
    print("\n🔄 Ejecutando evaluación del modelo...")
    
    # Obtener predicciones
    y_true, y_pred, y_proba = get_predictions(model, data_loader, device)
    
    # Calcular métricas
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    # Crear directorio si no existe
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Generar visualizaciones
    if plot:
        # Matriz de confusión
        cm_path = f"{save_dir}/confusion_matrix.png" if save_dir else None
        plot_confusion_matrix(y_true, y_pred, cm_path)
        
        # Curva ROC
        roc_path = f"{save_dir}/roc_curve.png" if save_dir else None
        plot_roc_curve(y_true, y_proba, roc_path)
        
        # Curva PR
        pr_path = f"{save_dir}/precision_recall_curve.png" if save_dir else None
        plot_precision_recall_curve(y_true, y_proba, pr_path)
    
    # Imprimir reporte
    print_evaluation_report(metrics)
    
    return metrics


# =============================================================================
# Test del módulo
# =============================================================================
if __name__ == "__main__":
    # Test con datos sintéticos
    print("Test del módulo de evaluación")
    print("-" * 40)
    
    # Simular predicciones
    np.random.seed(42)
    n_samples = 200
    
    # Crear datos de prueba
    y_true = np.array([0] * 100 + [1] * 100)  # 100 normal, 100 disruptivo
    
    # Simular un modelo bueno (90% correcto)
    y_pred = y_true.copy()
    # Introducir algunos errores
    errors = np.random.choice(n_samples, size=20, replace=False)
    y_pred[errors] = 1 - y_pred[errors]
    
    # Simular probabilidades
    y_proba = np.where(y_true == 1, 
                       np.random.uniform(0.6, 1.0, n_samples),
                       np.random.uniform(0.0, 0.4, n_samples))
    
    # Calcular métricas
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    # Imprimir reporte
    print_evaluation_report(metrics)
    
    print("\n✅ Módulo de evaluación funciona correctamente")