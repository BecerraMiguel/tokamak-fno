"""
scripts/train_fno.py

Entrenamiento del Fourier Neural Operator para predicción de disrupciones.

Día 9 del plan de implementación.

Configuración:
- in_channels: 5
- hidden_channels: 32
- modes: 16
- n_layers: 3
- lr: 1e-3
- epochs: 30
- batch_size: 32

Uso:
    python scripts/train_fno.py
"""

import sys
sys.path.insert(0, '.')

import os
import json
import torch
import torch.nn as nn
from datetime import datetime

from src.data.loader import get_dataloaders
from src.models.fno import FNO1d
from src.training.train import train_model
from src.training.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve


def main():
    # =========================================
    # Configuración
    # =========================================
    config = {
        # Modelo FNO
        'in_channels': 5,
        'out_channels': 2,
        'hidden_channels': 32,
        'modes': 16,
        'n_layers': 3,
        
        # Entrenamiento
        'learning_rate': 1e-3,
        'epochs': 30,
        'batch_size': 32,
        'weight_decay': 1e-4,
        
        # Early stopping
        'patience': 10,
        
        # Datos
        'data_path': 'data/tokamak_synthetic_1000.h5',
        'val_split': 0.2,
        'seed': 42,
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=" * 60)
    print("🔬 FNO Training - Día 9")
    print("=" * 60)
    print(f"\nConfiguración:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # =========================================
    # Cargar datos
    # =========================================
    print("📊 Cargando datos...")
    train_loader, val_loader, info = get_dataloaders(
        h5_path=config['data_path'],
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        seed=config['seed']
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Verificar shape de datos
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"  Input shape: {sample_batch.shape}")
    print(f"  Labels shape: {sample_labels.shape}")
    
    # =========================================
    # Crear modelo FNO
    # =========================================
    print("\n🧠 Creando modelo FNO...")
    model = FNO1d(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        hidden_channels=config['hidden_channels'],
        modes=config['modes'],
        n_layers=config['n_layers']
    )
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros totales: {total_params:,}")
    print(f"  Parámetros entrenables: {trainable_params:,}")
    
    # Mover a device
    device = torch.device(config['device'])
    model = model.to(device)
    print(f"  Device: {device}")
    
    # =========================================
    # Configurar entrenamiento
    # =========================================
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler: reduce LR si val_loss no mejora
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # =========================================
    # Entrenar
    # =========================================
    print("\n🏋️ Iniciando entrenamiento...")
    print("-" * 60)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['epochs'],
        device=device,
        patience=config['patience'],
        model_name='fno'
    )
    
    # =========================================
    # Evaluar
    # =========================================
    print("\n📈 Evaluando modelo...")
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load('results/best_fno.pt'))
    model.eval()
    
    # Evaluar con el módulo actualizado
    metrics = evaluate_model(
        model=model, 
        data_loader=val_loader, 
        device=device,
        save_dir='results',
        plot=True,
        model_name='fno'
    )
    
    print("\n" + "=" * 60)
    print("📊 RESULTADOS FNO")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  TPR:       {metrics['tpr']:.4f}")
    print(f"  FPR:       {metrics['fpr']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    
    # =========================================
    # Guardar resultados
    # =========================================
    print("\n💾 Guardando resultados...")
    
    # Crear directorio si no existe
    os.makedirs('results', exist_ok=True)
    
    # Preparar métricas para JSON (convertir arrays a listas si es necesario)
    metrics_for_json = {}
    for key, value in metrics.items():
        if hasattr(value, 'tolist'):
            metrics_for_json[key] = value.tolist()
        else:
            metrics_for_json[key] = value
    
    # Guardar métricas
    results = {
        'config': config,
        'metrics': metrics_for_json,
        'history': history,
        'total_params': total_params,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('results/fno_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("  ✅ Métricas guardadas en results/fno_results.json")
    print("  ✅ Confusion matrix guardada en results/fno_confusion_matrix.png")
    print("  ✅ ROC curve guardada en results/fno_roc_curve.png")
    
    # =========================================
    # Comparación con Baseline
    # =========================================
    print("\n" + "=" * 60)
    print("📊 COMPARACIÓN FNO vs BASELINE")
    print("=" * 60)
    
    # Intentar cargar resultados del baseline
    try:
        with open('results/baseline_results.json', 'r') as f:
            baseline_results = json.load(f)
        
        baseline_metrics = baseline_results.get('metrics', {})
        
        print(f"\n{'Métrica':<15} {'Baseline':>12} {'FNO':>12} {'Diferencia':>12}")
        print("-" * 55)
        
        for metric_name in ['accuracy', 'tpr', 'fpr', 'precision', 'f1', 'auc']:
            # Manejar tanto 'auc' como 'roc_auc'
            if metric_name == 'auc':
                baseline_val = baseline_metrics.get('auc', baseline_metrics.get('roc_auc', 0))
                fno_val = metrics.get('auc', metrics.get('roc_auc', 0))
            else:
                baseline_val = baseline_metrics.get(metric_name, 0)
                fno_val = metrics.get(metric_name, 0)
            
            diff = fno_val - baseline_val
            
            # Formato especial para FPR (menor es mejor)
            if metric_name == 'fpr':
                indicator = "✅" if diff <= 0 else "⚠️"
            else:
                indicator = "✅" if diff >= 0 else "⚠️"
            
            print(f"{metric_name:<15} {baseline_val:>12.4f} {fno_val:>12.4f} {diff:>+12.4f} {indicator}")
        
    except FileNotFoundError:
        print("  ⚠️ No se encontró results/baseline_results.json")
        print("  Ejecuta primero el entrenamiento del baseline para comparar.")
    
    print("\n" + "=" * 60)
    print("✅ Entrenamiento FNO completado!")
    print("=" * 60)
    
    return metrics, history


if __name__ == "__main__":
    main()
