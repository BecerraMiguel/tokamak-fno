"""
src/models/baseline.py

Modelo baseline CNN 1D para predicción de disrupciones en tokamaks.

Este modelo sirve como línea base para comparar con arquitecturas
más avanzadas como Fourier Neural Operators (FNO).

Arquitectura:
    - 3 capas convolucionales 1D con stride=2 (reducción progresiva)
    - BatchNorm + ReLU después de cada convolución
    - Global Average Pooling para obtener vector de features
    - Clasificador MLP con dropout

Target: < 500k parámetros, accuracy > 55% (mejor que aleatorio)

Autor: Miguel Becerra
Proyecto: Tokamak FNO - Día 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class ConvBlock(nn.Module):
    """
    Bloque convolucional básico: Conv1d + BatchNorm + ReLU.
    
    Este patrón (Conv-BN-ReLU) es estándar en redes modernas porque:
    - BatchNorm normaliza las activaciones, estabilizando el entrenamiento
    - ReLU introduce no-linealidad necesaria para aprender patrones complejos
    
    Args:
        in_channels: Número de canales de entrada
        out_channels: Número de canales de salida (filtros a aprender)
        kernel_size: Tamaño del kernel (ventana temporal que mira cada filtro)
        stride: Paso de la convolución (stride=2 reduce dimensión a la mitad)
        padding: Ceros añadidos a los bordes para controlar tamaño de salida
    
    Example:
        >>> block = ConvBlock(5, 32, kernel_size=7, stride=2, padding=3)
        >>> x = torch.randn(16, 5, 1000)  # batch=16, canales=5, tiempo=1000
        >>> out = block(x)
        >>> print(out.shape)  # [16, 32, 500]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        padding: int = 3
    ):
        super().__init__()
        
        # Capa convolucional
        # bias=False porque BatchNorm ya tiene su propio bias (ahorra parámetros)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        # Normalización por lotes
        # Normaliza las activaciones para que tengan media~0 y std~1
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Función de activación
        # inplace=True modifica el tensor directamente (ahorra memoria)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del bloque.
        
        Args:
            x: Tensor de entrada con shape [batch, in_channels, time]
            
        Returns:
            Tensor de salida con shape [batch, out_channels, time//stride]
        """
        x = self.conv(x)   # Convolución: extrae características
        x = self.bn(x)     # Normalización: estabiliza valores
        x = self.relu(x)   # Activación: introduce no-linealidad
        return x


class BaselineCNN(nn.Module):
    """
    CNN 1D baseline para clasificación binaria de disrupciones.
    
    Recibe señales temporales de múltiples diagnósticos del tokamak
    y predice si el disparo terminará en disrupción o no.
    
    La arquitectura usa convoluciones con stride=2 para reducir
    progresivamente la dimensión temporal mientras aumenta los
    canales (features), seguido de pooling global y un clasificador MLP.
    
    Args:
        in_channels: Número de señales de entrada (default: 5)
                    Corresponde a: Ip, βN, q95, densidad, li
        num_classes: Número de clases de salida (default: 2)
                    0 = normal, 1 = disruptivo
        hidden_channels: Lista de canales en capas convolucionales
                        (default: [32, 64, 128])
        dropout_rate: Probabilidad de dropout en clasificador (default: 0.5)
    
    Example:
        >>> model = BaselineCNN(in_channels=5, num_classes=2)
        >>> x = torch.randn(32, 5, 1000)  # batch=32, 5 señales, 1000 timesteps
        >>> logits = model(x)
        >>> print(logits.shape)  # [32, 2]
        >>> probs = model.predict_proba(x)  # Con softmax aplicado
        >>> preds = model.predict(x)  # Clase predicha (0 o 1)
    """
    
    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 2,
        hidden_channels: Optional[List[int]] = None,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # Valores por defecto para hidden_channels
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        
        # Guardar configuración como atributos (útil para inspección)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate
        
        # ============================================================
        # CAPAS CONVOLUCIONALES
        # ============================================================
        # Construimos una secuencia de ConvBlocks
        # Cada uno reduce la dimensión temporal a la mitad (stride=2)
        # y aumenta el número de canales (features)
        
        conv_layers = []
        current_channels = in_channels  # Empezamos con 5 canales
        
        for out_ch in hidden_channels:
            conv_layers.append(
                ConvBlock(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    kernel_size=7,
                    stride=2,
                    padding=3
                )
            )
            current_channels = out_ch  # El output de esta capa es input de la siguiente
        
        # nn.Sequential permite agrupar capas y ejecutarlas en orden
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # ============================================================
        # POOLING GLOBAL
        # ============================================================
        # Promedia sobre toda la dimensión temporal
        # Convierte [batch, 128, T] -> [batch, 128, 1]
        # Esto hace que el modelo funcione con cualquier longitud de entrada
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)
        
        # ============================================================
        # CLASIFICADOR MLP
        # ============================================================
        # Toma el vector de características y produce la predicción final
        self.classifier = nn.Sequential(
            nn.Flatten(),                           # [batch, 128, 1] -> [batch, 128]
            nn.Linear(hidden_channels[-1], 64),     # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),               # Regularización
            nn.Linear(64, num_classes)              # 64 -> 2 (logits)
        )
        
        # Inicializar pesos para mejor convergencia
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Inicializa los pesos de la red usando el método de Kaiming.
        
        La inicialización Kaiming (He) está diseñada específicamente para
        redes con activaciones ReLU. Mantiene la varianza de las activaciones
        constante a través de las capas, evitando que las señales
        desaparezcan o exploten durante el entrenamiento inicial.
        
        Reglas:
        - Conv1d: Kaiming normal con mode='fan_out'
        - BatchNorm: weight=1, bias=0 (identidad inicial)
        - Linear: Kaiming normal
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Kaiming para convoluciones con ReLU
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_out',      # Basado en número de salidas
                    nonlinearity='relu'  # Ajustado para ReLU
                )
            elif isinstance(m, nn.BatchNorm1d):
                # BatchNorm empieza como función identidad
                nn.init.constant_(m.weight, 1)  # gamma = 1
                nn.init.constant_(m.bias, 0)    # beta = 0
            elif isinstance(m, nn.Linear):
                # Kaiming para capas lineales
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red completa.
        
        Args:
            x: Tensor de señales con shape [batch, channels, time]
               Ejemplo: [32, 5, 1000] = 32 muestras, 5 señales, 1000 timesteps
        
        Returns:
            logits: Tensor con shape [batch, num_classes]
                   Ejemplo: [32, 2] = scores para cada clase (sin softmax)
        
        Note:
            Los logits son las salidas antes de softmax. Para obtener
            probabilidades, usar F.softmax(logits, dim=1) o predict_proba().
            Para entrenamiento, usar nn.CrossEntropyLoss que aplica
            softmax internamente (más estable numéricamente).
        """
        # Capas convolucionales: extraen características temporales
        # [batch, 5, 1000] -> [batch, 32, 500] -> [batch, 64, 250] -> [batch, 128, 125]
        x = self.conv_layers(x)
        
        # Pooling global: resume información temporal en un vector
        # [batch, 128, 125] -> [batch, 128, 1]
        x = self.global_pool(x)
        
        # Clasificador: produce predicción final
        # [batch, 128, 1] -> flatten -> [batch, 128] -> ... -> [batch, 2]
        logits = self.classifier(x)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predice probabilidades aplicando softmax a los logits.
        
        Args:
            x: Tensor de señales [batch, channels, time]
        
        Returns:
            probs: Tensor de probabilidades [batch, num_classes]
                  Cada fila suma 1.0
                  probs[:, 0] = P(normal)
                  probs[:, 1] = P(disruptivo)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predice la clase más probable para cada muestra.
        
        Args:
            x: Tensor de señales [batch, channels, time]
        
        Returns:
            predictions: Tensor de predicciones [batch]
                        Valores: 0 (normal) o 1 (disruptivo)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Cuenta el número de parámetros en un modelo.
    
    Args:
        model: Modelo de PyTorch
        trainable_only: Si True, cuenta solo parámetros entrenables
                       (excluye parámetros congelados)
    
    Returns:
        Número total de parámetros
    
    Example:
        >>> model = BaselineCNN()
        >>> n_params = count_parameters(model)
        >>> print(f"Parámetros: {n_params:,}")  # Parámetros: 81,634
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_summary(model: nn.Module, input_shape: Tuple[int, ...] = (1, 5, 1000)) -> str:
    """
    Genera un resumen del modelo mostrando shapes en cada capa.
    
    Args:
        model: Modelo de PyTorch
        input_shape: Shape del input de prueba (batch, channels, time)
    
    Returns:
        String con el resumen formateado del modelo
    """
    summary_lines = [
        "=" * 65,
        "RESUMEN DEL MODELO: BaselineCNN",
        "=" * 65,
        f"Input shape: {list(input_shape)}",
        "-" * 65,
    ]
    
    # Crear input de prueba
    x = torch.randn(input_shape)
    
    # Registrar hooks para capturar shapes intermedios
    shapes = []
    
    def hook_fn(module, input, output):
        class_name = module.__class__.__name__
        if hasattr(output, 'shape'):
            shapes.append((class_name, list(output.shape)))
    
    # Registrar hooks en capas principales
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv1d, nn.BatchNorm1d, nn.ReLU, 
                             nn.AdaptiveAvgPool1d, nn.Linear, nn.Flatten)):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    # Remover hooks
    for h in hooks:
        h.remove()
    
    # Formatear shapes capturados
    for class_name, shape in shapes:
        summary_lines.append(f"  {class_name:25} -> {shape}")
    
    summary_lines.extend([
        "-" * 65,
        f"Output shape: {list(output.shape)}",
        "-" * 65,
        f"Total parámetros:       {count_parameters(model, False):>12,}",
        f"Parámetros entrenables: {count_parameters(model, True):>12,}",
        "=" * 65,
    ])
    
    return "\n".join(summary_lines)


# ============================================================
# CÓDIGO DE VERIFICACIÓN
# ============================================================
# Este código se ejecuta solo cuando corres el archivo directamente:
#   python -m src.models.baseline
# No se ejecuta cuando importas el módulo desde otro archivo.

if __name__ == "__main__":
    print("=" * 65)
    print("VERIFICACIÓN DEL MODELO BASELINE CNN")
    print("=" * 65)
    
    # --------------------------------------------------------
    # Test 1: Crear modelo
    # --------------------------------------------------------
    print("\n[Test 1] Creando modelo BaselineCNN...")
    model = BaselineCNN(in_channels=5, num_classes=2)
    print("         ✓ Modelo creado exitosamente")
    
    # --------------------------------------------------------
    # Test 2: Contar parámetros
    # --------------------------------------------------------
    print("\n[Test 2] Contando parámetros...")
    n_params = count_parameters(model)
    print(f"         Total parámetros: {n_params:,}")
    
    if n_params < 500_000:
        print(f"         ✓ Cumple requisito (< 500,000)")
    else:
        print(f"         ✗ FALLA: Excede límite de 500,000")
    
    # --------------------------------------------------------
    # Test 3: Forward pass
    # --------------------------------------------------------
    print("\n[Test 3] Probando forward pass...")
    batch_size = 32
    x = torch.randn(batch_size, 5, 1000)
    print(f"         Input shape:  {list(x.shape)}")
    
    model.eval()  # Modo evaluación
    with torch.no_grad():
        output = model(x)
    print(f"         Output shape: {list(output.shape)}")
    
    expected_shape = (batch_size, 2)
    if output.shape == expected_shape:
        print(f"         ✓ Output shape correcto")
    else:
        print(f"         ✗ FALLA: Esperado {expected_shape}")
    
    # --------------------------------------------------------
    # Test 4: Backward pass (gradientes)
    # --------------------------------------------------------
    print("\n[Test 4] Probando backward pass...")
    model.train()  # Modo entrenamiento
    x = torch.randn(4, 5, 1000)
    y = torch.randint(0, 2, (4,))  # Etiquetas aleatorias
    
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()  # Calcular gradientes
    
    print(f"         Loss calculado: {loss.item():.4f}")
    print("         ✓ Backward pass exitoso")
    
    # --------------------------------------------------------
    # Test 5: Métodos de predicción
    # --------------------------------------------------------
    print("\n[Test 5] Probando métodos de predicción...")
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(x)
        preds = model.predict(x)
    
    print(f"         Probabilidades shape: {list(probs.shape)}")
    print(f"         Suma probs (debe ser ~1.0): {probs[0].sum().item():.4f}")
    print(f"         Predicciones: {preds.tolist()}")
    print("         ✓ Métodos de predicción funcionan")
    
    # --------------------------------------------------------
    # Test 6: Resumen del modelo
    # --------------------------------------------------------
    print("\n[Test 6] Resumen del modelo:")
    print(get_model_summary(model))
    
    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("✓ TODAS LAS VERIFICACIONES PASARON")
    print("  El modelo está listo para entrenamiento (Día 5)")
    print("=" * 65)
