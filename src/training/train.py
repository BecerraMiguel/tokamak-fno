"""
src/training/train.py

Funciones de entrenamiento para predicción de disrupciones en tokamaks.

Incluye:
    - train_epoch: Una época de entrenamiento
    - evaluate: Evaluación en conjunto de validación
    - train_model: Loop completo de entrenamiento
    - EarlyStopping: Detención temprana para evitar overfitting

Autor: Miguel Becerra
Proyecto: Tokamak FNO 
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from pathlib import Path
import time


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Ejecuta UNA época de entrenamiento.
    
    Una época = una pasada completa por todo el dataset de entrenamiento.
    
    Args:
        model: El modelo a entrenar (BaselineCNN o FNO1d)
        dataloader: DataLoader con datos de entrenamiento
        criterion: Función de pérdida (CrossEntropyLoss)
        optimizer: Optimizador (Adam)
        device: 'cuda' o 'cpu'
    
    Returns:
        Tuple[float, float]: (loss promedio, accuracy)
    
    Proceso por cada batch:
        1. Mover datos a GPU/CPU
        2. Forward pass: obtener predicciones
        3. Calcular loss
        4. Backward pass: calcular gradientes
        5. Optimizer step: actualizar pesos
        6. Limpiar gradientes
    """
    # Poner modelo en modo entrenamiento
    # Esto activa Dropout y BatchNorm en modo training
    model.train()
    
    # Acumuladores para métricas
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Iterar sobre todos los batches
    for batch_idx, (data, targets) in enumerate(dataloader):
        # 1. Mover datos al dispositivo (GPU/CPU)
        data = data.to(device)        # Shape: [batch, 5, 1000]
        targets = targets.to(device)  # Shape: [batch]
        
        # 2. Limpiar gradientes del paso anterior
        # IMPORTANTE: Si no hacemos esto, los gradientes se acumulan
        optimizer.zero_grad()
        
        # 3. Forward pass: calcular predicciones
        outputs = model(data)  # Shape: [batch, 2] (logits para cada clase)
        
        # 4. Calcular loss
        loss = criterion(outputs, targets)
        
        # 5. Backward pass: calcular gradientes
        # PyTorch calcula automáticamente ∂Loss/∂w para cada peso
        loss.backward()
        
        # 6. Actualizar pesos usando los gradientes
        # w_nuevo = w_viejo - learning_rate × gradiente
        optimizer.step()
        
        # Acumular métricas
        total_loss += loss.item() * data.size(0)  # Multiplicar por batch size
        
        # Calcular accuracy
        _, predicted = torch.max(outputs, dim=1)  # Índice de clase con mayor logit
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    
    # Calcular promedios
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


@torch.no_grad()  # Decorador que desactiva cálculo de gradientes
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evalúa el modelo en un conjunto de datos (validación o test).
    
    IMPORTANTE: Usamos @torch.no_grad() porque:
    1. No necesitamos gradientes para evaluar
    2. Ahorra memoria GPU
    3. Es más rápido
    
    Args:
        model: Modelo a evaluar
        dataloader: DataLoader con datos de validación/test
        criterion: Función de pérdida
        device: 'cuda' o 'cpu'
    
    Returns:
        Tuple[float, float]: (loss promedio, accuracy)
    """
    # Poner modelo en modo evaluación
    # Esto desactiva Dropout y pone BatchNorm en modo inference
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for data, targets in dataloader:
        data = data.to(device)
        targets = targets.to(device)
        
        # Solo forward pass, sin backward
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * data.size(0)
        
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


class EarlyStopping:
    """
    Detención temprana para evitar overfitting.
    
    Monitorea la pérdida de validación y detiene el entrenamiento
    si no mejora después de 'patience' épocas consecutivas.
    
    ¿Por qué es importante?
    - El modelo puede empezar a memorizar datos de entrenamiento
    - La pérdida de entrenamiento sigue bajando
    - Pero la pérdida de validación empieza a SUBIR
    - Esto es overfitting → detenemos antes
    
    Ejemplo de uso:
        early_stopping = EarlyStopping(patience=5)
        for epoch in range(100):
            train_loss = train_epoch(...)
            val_loss, _ = evaluate(...)
            
            if early_stopping(val_loss, model):
                print("Early stopping triggered!")
                break
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        Args:
            patience: Número de épocas a esperar sin mejora antes de detener
            min_delta: Mejora mínima para considerar que hubo progreso
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Llamar después de cada época de validación.
        
        Returns:
            True si debemos detener el entrenamiento
        """
        if self.best_loss is None:
            # Primera época
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss < self.best_loss - self.min_delta:
            # ¡Mejora! Resetear contador
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            # No mejoró
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def load_best_model(self, model: nn.Module):
        """Carga los pesos del mejor modelo encontrado."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    num_epochs: Optional[int] = None,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    save_dir: str = "results",
    patience: int = 7,
    model_name: str = "model",
    verbose: bool = True
) -> Dict:
    """
    Loop completo de entrenamiento con validación y guardado de mejor modelo.
    
    Soporta dos modos de uso:
    
    Modo 1 - Simple (crea criterion/optimizer internamente):
        history = train_model(model, train_loader, val_loader, epochs=30)
    
    Modo 2 - Avanzado (pasa criterion/optimizer/scheduler externos):
        history = train_model(model, train_loader, val_loader,
                              criterion=criterion, optimizer=optimizer,
                              scheduler=scheduler, num_epochs=30,
                              model_name='fno')
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        criterion: Función de pérdida (opcional, default: CrossEntropyLoss)
        optimizer: Optimizador (opcional, default: Adam)
        scheduler: Learning rate scheduler (opcional)
        num_epochs: Número de épocas (alias para epochs)
        epochs: Número máximo de épocas (default: 30)
        learning_rate: Tasa de aprendizaje para Adam (si no se pasa optimizer)
        device: Dispositivo de cómputo (auto-detecta si None)
        save_dir: Directorio para guardar modelo y métricas
        patience: Épocas para early stopping
        model_name: Nombre base para guardar el modelo (default: 'model')
        verbose: Si True, imprime progreso
    
    Returns:
        Dict con historial de entrenamiento:
            - train_losses: Lista de pérdidas de entrenamiento
            - val_losses: Lista de pérdidas de validación
            - train_accs: Lista de accuracies de entrenamiento
            - val_accs: Lista de accuracies de validación
            - best_val_acc: Mejor accuracy de validación
            - best_epoch: Época del mejor modelo
    """
    # Resolver num_epochs vs epochs (num_epochs tiene prioridad)
    total_epochs = num_epochs if num_epochs is not None else epochs
    
    # Auto-detectar dispositivo
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Entrenando en: {device}")
        print(f"Épocas máximas: {total_epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"Modelo se guardará como: best_{model_name}.pt")
        print("-" * 50)
    
    # Mover modelo al dispositivo
    model = model.to(device)
    
    # Configurar loss si no se proporciona
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Configurar optimizer si no se proporciona
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if verbose:
            print(f"Learning rate: {learning_rate}")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Crear directorio de resultados
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Historial de métricas
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    # Timer
    start_time = time.time()
    
    # Loop principal de entrenamiento
    for epoch in range(total_epochs):
        epoch_start = time.time()
        
        # Entrenar una época
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluar en validación
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        
        # Actualizar scheduler si existe (ReduceLROnPlateau necesita val_loss)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Guardar métricas
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)
        
        # Verificar si es el mejor modelo
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch + 1
            
            # Guardar mejor modelo con nombre personalizado
            model_filename = f'best_{model_name}.pt'
            torch.save(model.state_dict(), save_path / model_filename)
        
        epoch_time = time.time() - epoch_start
        
        # Imprimir progreso
        if verbose:
            print(f"Época {epoch+1:3d}/{total_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Tiempo: {epoch_time:.1f}s")
        
        # Verificar early stopping
        if early_stopping(val_loss, model):
            if verbose:
                print(f"\nEarly stopping en época {epoch+1}")
                print(f"Mejor validación fue en época {history['best_epoch']}")
            break
    
    total_time = time.time() - start_time
    
    if verbose:
        print("-" * 50)
        print(f"Entrenamiento completado en {total_time:.1f} segundos")
        print(f"Mejor accuracy de validación: {history['best_val_acc']:.4f} "
              f"(época {history['best_epoch']})")
        print(f"Modelo guardado en: {save_path / f'best_{model_name}.pt'}")
    
    # Cargar mejor modelo al final
    early_stopping.load_best_model(model)
    
    return history


if __name__ == "__main__":
    """
    Script de prueba para verificar que el módulo funciona.
    NOTA: Este test es solo para verificar sintaxis.
    El entrenamiento real se hará en Google Colab con GPU.
    """
    print("=" * 60)
    print("TEST: Verificación de sintaxis del módulo de entrenamiento")
    print("=" * 60)
    
    # Verificar imports
    print("\n1. Verificando imports...")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.models.baseline import BaselineCNN
        from src.data.loader import get_dataloaders
        print("   ✓ Imports correctos")
    except ImportError as e:
        print(f"   ✗ Error de import: {e}")
        sys.exit(1)
    
    # Verificar que las funciones existen
    print("\n2. Verificando funciones...")
    print(f"   ✓ train_epoch: {callable(train_epoch)}")
    print(f"   ✓ evaluate: {callable(evaluate)}")
    print(f"   ✓ train_model: {callable(train_model)}")
    print(f"   ✓ EarlyStopping: {callable(EarlyStopping)}")
    
    # Test rápido de EarlyStopping
    print("\n3. Test de EarlyStopping...")
    es = EarlyStopping(patience=3)
    
    # Simular mejoras
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
    
    dummy = DummyModel()
    
    # Simular épocas: loss baja, luego sube
    test_losses = [0.7, 0.6, 0.5, 0.55, 0.56, 0.57, 0.58]
    for i, loss in enumerate(test_losses):
        should_stop = es(loss, dummy)
        status = "STOP" if should_stop else "continuar"
        print(f"   Época {i+1}: loss={loss:.2f} -> {status}")
        if should_stop:
            break
    
    print("\n" + "=" * 60)
    print("✓ Módulo de entrenamiento verificado")
    print("  El entrenamiento real se hará en Google Colab")
    print("=" * 60)
