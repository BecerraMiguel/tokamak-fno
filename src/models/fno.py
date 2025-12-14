"""
src/models/fno.py

Fourier Neural Operator para predicción de disrupciones en tokamaks.

Implementación basada en:
- Li et al. (2020) "Fourier Neural Operator for Parametric PDEs"

La arquitectura FNO aprende operadores en el espacio de Fourier,
permitiendo:
1. Captura de patrones globales (no solo locales como CNN)
2. Independencia de resolución temporal
3. Transfer learning entre dispositivos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """
    Convolución espectral 1D - el bloque fundamental del FNO.
    
    Opera en el espacio de Fourier:
    1. FFT de la entrada al dominio de frecuencias
    2. Multiplicación por pesos complejos (solo modos bajos)
    3. IFFT de vuelta al dominio temporal
    
    Args:
        in_channels: Número de canales de entrada
        out_channels: Número de canales de salida
        modes: Número de modos de Fourier a mantener (frecuencias bajas)
    
    Shapes:
        Input:  [batch, in_channels, time]
        Output: [batch, out_channels, time]
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Número de modos de Fourier a usar
        
        # Escala para inicialización (similar a Xavier)
        self.scale = 1 / (in_channels * out_channels)
        
        # Pesos complejos para la transformación espectral
        # Shape: [in_channels, out_channels, modes]
        # Usamos números complejos porque FFT produce coeficientes complejos
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, 
                                    dtype=torch.cfloat)
        )
    
    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Multiplicación compleja por lotes.
        
        Realiza: output[b,o,m] = sum_i(input[b,i,m] * weights[i,o,m])
        
        Args:
            input: [batch, in_channels, modes] - coeficientes de Fourier
            weights: [in_channels, out_channels, modes] - pesos aprendibles
        
        Returns:
            output: [batch, out_channels, modes]
        """
        # Usamos einsum para la contracción de tensores
        # 'bim,iom->bom' significa:
        #   b = batch, i = in_channels, o = out_channels, m = modes
        return torch.einsum('bim,iom->bom', input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la convolución espectral.
        
        Args:
            x: [batch, in_channels, time] - señal en dominio temporal
        
        Returns:
            [batch, out_channels, time] - señal transformada
        """
        batch_size = x.shape[0]
        time_steps = x.shape[-1]
        
        # 1. FFT: Transformar al espacio de Fourier
        # rfft = FFT real, retorna solo frecuencias positivas (por simetría)
        # Shape: [batch, in_channels, time//2 + 1] (complejos)
        x_ft = torch.fft.rfft(x)
        
        # 2. Preparar tensor de salida (inicializado en ceros)
        # Necesitamos el tamaño correcto para la IFFT posterior
        out_ft = torch.zeros(
            batch_size, 
            self.out_channels, 
            time_steps // 2 + 1,  # Tamaño de rfft output
            dtype=torch.cfloat,
            device=x.device
        )
        
        # 3. Multiplicar solo los primeros 'modes' coeficientes
        # Esto implementa el "mode truncation" - solo frecuencias bajas
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes],  # Solo primeros 'modes' coeficientes
            self.weights
        )
        
        # 4. IFFT: Transformar de vuelta al espacio temporal
        # irfft = IFFT real, reconstruye señal real desde coeficientes complejos
        x = torch.fft.irfft(out_ft, n=time_steps)
        
        return x
    
class FourierLayer(nn.Module):
    """
    Una capa completa del FNO con conexión residual.
    
    Combina:
    1. SpectralConv1d: Transformación en espacio de Fourier (patrones globales)
    2. Conv1d: Transformación local (bypass path)
    3. Conexión residual: Estabiliza el entrenamiento
    
    La fórmula es:
        output = activation(SpectralConv(x) + Conv(x))
    
    Args:
        channels: Número de canales (entrada = salida en esta capa)
        modes: Número de modos de Fourier
        activation: Función de activación (default: GELU)
    """
    
    def __init__(self, channels: int, modes: int, activation: str = 'gelu'):
        super().__init__()
        
        # Rama espectral: captura patrones globales en frecuencia
        self.spectral_conv = SpectralConv1d(channels, channels, modes)
        
        # Rama local: Conv1d 1x1 actúa como transformación lineal por punto
        # Esto es el "bypass" o "skip path" que preserva información local
        self.conv = nn.Conv1d(channels, channels, kernel_size=1)
        
        # Normalización por lotes para estabilizar entrenamiento
        self.bn = nn.BatchNorm1d(channels)
        
        # Función de activación
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la capa Fourier.
        
        Args:
            x: [batch, channels, time]
        
        Returns:
            [batch, channels, time]
        """
        # Rama espectral (global)
        x_spectral = self.spectral_conv(x)
        
        # Rama local (bypass)
        x_local = self.conv(x)
        
        # Combinar ambas ramas
        x = x_spectral + x_local
        
        # Normalización y activación
        x = self.bn(x)
        x = self.activation(x)
        
        return x
    
class FNO1d(nn.Module):
    """
    Fourier Neural Operator 1D para clasificación de disrupciones.
    
    Arquitectura:
        Input → Lifting → N×FourierLayers → Projection → Classifier → Output
    
    Args:
        in_channels: Canales de entrada (5 para señales de tokamak)
        out_channels: Clases de salida (2: normal vs disruptivo)
        hidden_channels: Canales en capas ocultas (default: 32)
        modes: Modos de Fourier a mantener (default: 16)
        n_layers: Número de capas Fourier (default: 3)
        activation: Función de activación (default: 'gelu')
    
    Shapes:
        Input:  [batch, in_channels, time]
        Output: [batch, out_channels]
    """
    
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 2,
        hidden_channels: int = 32,
        modes: int = 16,
        n_layers: int = 3,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.modes = modes
        self.n_layers = n_layers
        
        # === Lifting Layer ===
        # Proyecta de in_channels a hidden_channels
        # Expande la dimensionalidad para procesar en las capas Fourier
        self.lifting = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        
        # === Fourier Layers ===
        # Stack de N capas Fourier
        self.fourier_layers = nn.ModuleList([
            FourierLayer(hidden_channels, modes, activation)
            for _ in range(n_layers)
        ])
        
        # === Projection Layer ===
        # Proyecta a más canales antes de la clasificación
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels * 4, hidden_channels * 4, kernel_size=1)
        )
        
        # === Classifier Head ===
        # Global pooling + MLP para clasificación
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [batch, channels, 1]
            nn.Flatten(),             # [batch, channels]
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels * 2, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del FNO.
        
        Args:
            x: [batch, in_channels, time] - señales diagnósticas
        
        Returns:
            [batch, out_channels] - logits de clasificación
        """
        # Lifting: expandir canales
        x = self.lifting(x)
        
        # Fourier layers: procesar en espacio espectral
        for layer in self.fourier_layers:
            x = layer(x)
        
        # Projection: preparar para clasificación
        x = self.projection(x)
        
        # Classifier: pooling global + MLP
        x = self.classifier(x)
        
        return x


def count_fno_parameters(model: nn.Module) -> int:
    """Cuenta parámetros totales y entrenables del modelo."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}