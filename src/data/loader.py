"""
src/data/loader.py

DataLoader y preprocesamiento para datos de tokamak.
Convierte datos HDF5 en batches de PyTorch listos para entrenamiento.

Este módulo proporciona:
- TokamakDataset: Clase Dataset de PyTorch para cargar datos HDF5
- get_dataloaders: Función para crear DataLoaders de train/validation
- print_dataset_info: Utilidad para mostrar información del dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Tuple, Optional, Dict


class TokamakDataset(Dataset):
    """
    Dataset de PyTorch para datos de disrupciones en tokamaks.
    
    Carga datos desde archivo HDF5, aplica normalización Z-score,
    y prepara tensores para entrenamiento.
    
    La normalización Z-score transforma cada señal para que tenga
    media=0 y desviación estándar=1, lo cual ayuda al entrenamiento
    de redes neuronales.
    
    Attributes:
        signals: Tensor de señales [n_samples, n_channels, time_steps]
        labels: Tensor de etiquetas [n_samples] (0=normal, 1=disruptivo)
        normalize: Si True, aplica Z-score normalization
        mean: Media por canal (calculada o proporcionada)
        std: Desviación estándar por canal (calculada o proporcionada)
    
    Example:
        >>> dataset = TokamakDataset("data/tokamak_synthetic_1000.h5")
        >>> signal, label = dataset[0]
        >>> print(signal.shape)  # [5, 1000] -> 5 canales, 1000 timesteps
    """
    
    def __init__(
        self,
        h5_path: str,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        max_length: Optional[int] = None
    ):
        """
        Inicializa el dataset cargando datos del archivo HDF5.
        
        Args:
            h5_path: Ruta al archivo HDF5 con los datos
            normalize: Si True, aplica normalización Z-score por canal
            mean: Media pre-calculada (usar para validation/test set)
            std: Std pre-calculada (usar para validation/test set)
            max_length: Longitud temporal máxima (trunca si es necesario)
        """
        super().__init__()
        
        # Cargar datos del archivo HDF5
        self.signals, self.labels = self._load_h5(h5_path)
        
        # Ajustar longitud temporal si se especifica
        if max_length is not None and self.signals.shape[2] > max_length:
            # Tomar los últimos max_length timesteps (más relevantes para predicción)
            self.signals = self.signals[:, :, -max_length:]
        
        # Normalización Z-score
        self.normalize = normalize
        if normalize:
            if mean is None or std is None:
                # Calcular estadísticas del dataset
                self.mean, self.std = self._compute_statistics()
            else:
                self.mean = mean
                self.std = std
            
            # Aplicar normalización
            self.signals = self._apply_normalization()
        else:
            self.mean = None
            self.std = None
        
        # Convertir a tensores de PyTorch
        self.signals = torch.FloatTensor(self.signals)
        self.labels = torch.LongTensor(self.labels)
    
    def _load_h5(self, h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga datos desde archivo HDF5.
        
        El archivo HDF5 debe contener:
        - 'signals' o 'data': Array de señales diagnósticas
        - 'labels': Array de etiquetas (0 o 1)
        
        Returns:
            signals: Array de señales [n_samples, n_channels, time_steps]
            labels: Array de etiquetas [n_samples]
        """
        with h5py.File(h5_path, 'r') as f:
            # Manejar diferentes nombres para las señales
            if 'signals' in f.keys():
                signals = f['signals'][:]
            elif 'data' in f.keys():
                signals = f['data'][:]
            else:
                available_keys = list(f.keys())
                raise KeyError(
                    f"No se encontró 'signals' ni 'data' en el archivo HDF5. "
                    f"Datasets disponibles: {available_keys}"
                )
            
            labels = f['labels'][:]
        
        return signals, labels
    
    def _compute_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula media y desviación estándar por canal.
        
        La normalización por canal es importante porque cada señal
        diagnóstica tiene escalas muy diferentes:
        - Ip (corriente de plasma): ~1-2 MA
        - βN (beta normalizado): ~0-4 (adimensional)
        - q95 (factor de seguridad): ~2-10
        - ne (densidad electrónica): ~1e19 m^-3
        - li (inductancia interna): ~0.5-1.5
        
        Returns:
            mean: Media por canal [n_channels]
            std: Desviación estándar por canal [n_channels]
        """
        # signals shape: [n_samples, n_channels, time_steps]
        # Calculamos estadísticas sobre samples y tiempo, por canal
        mean = self.signals.mean(axis=(0, 2))  # [n_channels]
        std = self.signals.std(axis=(0, 2))    # [n_channels]
        
        # Evitar división por cero (si algún canal tiene varianza cero)
        std = np.where(std < 1e-8, 1.0, std)
        
        return mean, std
    
    def _apply_normalization(self) -> np.ndarray:
        """
        Aplica normalización Z-score: (x - mean) / std
        
        Después de esta transformación, cada canal tendrá
        aproximadamente media=0 y std=1.
        
        Returns:
            Señales normalizadas [n_samples, n_channels, time_steps]
        """
        # Expandir dimensiones para broadcasting
        # mean y std: [n_channels] -> [1, n_channels, 1]
        mean = self.mean[np.newaxis, :, np.newaxis]
        std = self.std[np.newaxis, :, np.newaxis]
        
        return (self.signals - mean) / std
    
    def __len__(self) -> int:
        """Retorna el número de muestras en el dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna una muestra del dataset.
        
        Args:
            idx: Índice de la muestra
            
        Returns:
            signal: Tensor de señales [n_channels, time_steps]
            label: Tensor escalar con la etiqueta (0 o 1)
        """
        return self.signals[idx], self.labels[idx]
    
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """
        Retorna las estadísticas de normalización.
        Útil para aplicar la misma normalización al validation set.
        
        Returns:
            Diccionario con 'mean' y 'std' arrays
        """
        return {'mean': self.mean, 'std': self.std}


def get_dataloaders(
    h5_path: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    max_length: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Crea DataLoaders para entrenamiento y validación.
    
    Esta función realiza los siguientes pasos:
    1. Carga el dataset completo desde el archivo HDF5
    2. Divide en train/validation de forma estratificada (misma proporción de clases)
    3. Calcula estadísticas de normalización SOLO del train set (evita data leakage)
    4. Aplica la misma normalización a ambos sets
    5. Crea DataLoaders con batching y shuffling
    
    ¿Por qué split estratificado?
    Si tienes 50% disparos disruptivos, tanto train como validation
    tendrán ~50% disruptivos. Esto asegura que las métricas de
    validation sean representativas.
    
    ¿Por qué normalizar solo con estadísticas de train?
    Usar datos de validation para calcular media/std sería "hacer trampa"
    porque estarías filtrando información del validation set al modelo.
    Esto se llama "data leakage" y hace que las métricas sean optimistas.
    
    Args:
        h5_path: Ruta al archivo HDF5 con los datos
        batch_size: Tamaño del batch para entrenamiento (default: 32)
        val_split: Fracción de datos para validación, entre 0 y 1 (default: 0.2)
        seed: Semilla para reproducibilidad (default: 42)
        num_workers: Número de workers para carga paralela (default: 0)
        max_length: Longitud temporal máxima (trunca si es necesario)
        
    Returns:
        train_loader: DataLoader para entrenamiento (con shuffle)
        val_loader: DataLoader para validación (sin shuffle)
        info: Diccionario con información del dataset
    
    Example:
        >>> train_loader, val_loader, info = get_dataloaders(
        ...     "data/tokamak_synthetic_1000.h5",
        ...     batch_size=32,
        ...     val_split=0.2
        ... )
        >>> for signals, labels in train_loader:
        ...     # signals.shape = [32, 5, 1000]
        ...     # labels.shape = [32]
        ...     pass
    """
    # Fijar semilla para reproducibilidad
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # =========================================
    # Cargar datos del archivo HDF5
    # =========================================
    with h5py.File(h5_path, 'r') as f:
        # Manejar diferentes nombres para las señales
        if 'signals' in f.keys():
            all_signals = f['signals'][:]
        elif 'data' in f.keys():
            all_signals = f['data'][:]
        else:
            available_keys = list(f.keys())
            raise KeyError(
                f"No se encontró 'signals' ni 'data' en el archivo HDF5. "
                f"Datasets disponibles: {available_keys}"
            )
        
        all_labels = f['labels'][:]
    
    n_samples = len(all_labels)
    
    # Ajustar longitud temporal si se especifica
    if max_length is not None and all_signals.shape[2] > max_length:
        all_signals = all_signals[:, :, -max_length:]
    
    # Crear índices y mezclar
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # =========================================
    # Split estratificado (mantener proporción de clases)
    # =========================================
    # Separar índices por clase
    idx_normal = indices[all_labels[indices] == 0]
    idx_disruptive = indices[all_labels[indices] == 1]
    
    # Calcular cuántas muestras van a validation de cada clase
    n_val_normal = int(len(idx_normal) * val_split)
    n_val_disruptive = int(len(idx_disruptive) * val_split)
    
    # Dividir cada clase
    val_indices = np.concatenate([
        idx_normal[:n_val_normal],
        idx_disruptive[:n_val_disruptive]
    ])
    train_indices = np.concatenate([
        idx_normal[n_val_normal:],
        idx_disruptive[n_val_disruptive:]
    ])
    
    # Separar datos según los índices
    train_signals = all_signals[train_indices]
    train_labels = all_labels[train_indices]
    val_signals = all_signals[val_indices]
    val_labels = all_labels[val_indices]
    
    # =========================================
    # Normalización Z-score
    # IMPORTANTE: Calcular estadísticas SOLO del train set
    # =========================================
    train_mean = train_signals.mean(axis=(0, 2))
    train_std = train_signals.std(axis=(0, 2))
    train_std = np.where(train_std < 1e-8, 1.0, train_std)  # Evitar división por cero
    
    # Normalizar ambos sets con estadísticas de train
    mean_expanded = train_mean[np.newaxis, :, np.newaxis]
    std_expanded = train_std[np.newaxis, :, np.newaxis]
    
    train_signals = (train_signals - mean_expanded) / std_expanded
    val_signals = (val_signals - mean_expanded) / std_expanded
    
    # =========================================
    # Crear datasets de PyTorch
    # =========================================
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_signals),
        torch.LongTensor(train_labels)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_signals),
        torch.LongTensor(val_labels)
    )
    
    # =========================================
    # Crear DataLoaders
    # =========================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # Mezclar datos en cada epoch (importante para training)
        num_workers=num_workers,
        pin_memory=True    # Acelera transferencia CPU->GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,     # NO mezclar validation (queremos resultados consistentes)
        num_workers=num_workers,
        pin_memory=True
    )
    
    # =========================================
    # Recopilar información del dataset
    # =========================================
    info = {
        'n_train': len(train_indices),
        'n_val': len(val_indices),
        'n_channels': all_signals.shape[1],
        'time_steps': all_signals.shape[2],
        'train_class_distribution': {
            'normal': int((train_labels == 0).sum()),
            'disruptive': int((train_labels == 1).sum())
        },
        'val_class_distribution': {
            'normal': int((val_labels == 0).sum()),
            'disruptive': int((val_labels == 1).sum())
        },
        'normalization': {
            'mean': train_mean,
            'std': train_std
        }
    }
    
    return train_loader, val_loader, info


def print_dataset_info(info: Dict) -> None:
    """
    Imprime información formateada del dataset.
    
    Args:
        info: Diccionario retornado por get_dataloaders()
    
    Example:
        >>> train_loader, val_loader, info = get_dataloaders(...)
        >>> print_dataset_info(info)
        ==================================================
        INFORMACIÓN DEL DATASET
        ==================================================
        ...
    """
    print("=" * 50)
    print("INFORMACIÓN DEL DATASET")
    print("=" * 50)
    print(f"\nMuestras de entrenamiento: {info['n_train']}")
    print(f"Muestras de validación:    {info['n_val']}")
    print(f"Canales (señales):         {info['n_channels']}")
    print(f"Pasos temporales:          {info['time_steps']}")
    print(f"\nDistribución de clases (Train):")
    print(f"  - Normal:     {info['train_class_distribution']['normal']}")
    print(f"  - Disruptivo: {info['train_class_distribution']['disruptive']}")
    print(f"\nDistribución de clases (Validation):")
    print(f"  - Normal:     {info['val_class_distribution']['normal']}")
    print(f"  - Disruptivo: {info['val_class_distribution']['disruptive']}")
    print("=" * 50)
