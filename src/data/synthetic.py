"""
Generador de datos sintéticos de tokamak para predicción de disrupciones.

Este módulo simula señales diagnósticas de un tokamak, incluyendo:
- Disparos normales (operación estable)
- Disparos disruptivos (con precursores físicamente realistas)

La física simulada incluye:
- Límites operacionales (Troyon, Greenwald, q95)
- Precursores MHD (oscilaciones crecientes)
- Secuencia de disrupción (thermal quench + current quench)

Señales generadas:
- ip: Corriente de plasma [MA]
- betan: Beta normalizado (adimensional)
- q95: Factor de seguridad en superficie 95% (adimensional)
- density: Densidad electrónica [10^19 m^-3]
- li: Inductancia interna (adimensional)

Autor: Proyecto Tokamak-FNO
Fecha: Día 2 del plan de implementación
"""

import numpy as np
import h5py
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import os
from tqdm import tqdm


@dataclass
class TokamakParameters:
    """
    Parámetros típicos de un tokamak tipo DIII-D.
    
    Física:
    -------
    - R0 (radio mayor): Distancia del eje del toro al centro del plasma.
      Para DIII-D es 1.67m, para ITER será 6.2m.
    
    - a (radio menor): Radio de la sección transversal del plasma.
      Define el volumen del plasma junto con R0.
    
    - B0 (campo toroidal): Campo magnético en el eje magnético.
      Campos más intensos permiten confinar plasma a mayor presión.
      β = presión_plasma / presión_magnética ∝ 1/B²
    
    Límites operacionales (de la física MHD):
    -----------------------------------------
    - Límite de Troyon: βN < 2.8-3.5 para evitar inestabilidades kink/ballooning
    - Límite de Greenwald: ne < Ip/(πa²) para evitar colapso radiativo
    - Límite de q95: q95 > 2 para evitar inestabilidad kink externa
    """
    # Geometría del tokamak (valores de DIII-D)
    R0: float = 1.67        # Radio mayor [m]
    a: float = 0.67         # Radio menor [m]
    B0: float = 2.0         # Campo toroidal [T]
    
    # Rangos operacionales típicos para operación normal
    Ip_range: Tuple[float, float] = (1.0, 2.0)      # Corriente plasma [MA]
    betan_range: Tuple[float, float] = (1.5, 2.5)   # Beta normalizado (lejos del límite ~3.5)
    q95_range: Tuple[float, float] = (3.0, 5.0)     # Factor de seguridad (lejos del límite ~2)
    density_range: Tuple[float, float] = (3.0, 8.0) # Densidad [10^19 m^-3]
    li_range: Tuple[float, float] = (0.8, 1.2)      # Inductancia interna
    
    # Límites físicos (umbrales de disrupción)
    troyon_limit: float = 3.5       # Límite de beta normalizado
    q95_critical: float = 2.0       # Factor de seguridad crítico
    greenwald_fraction_limit: float = 1.0  # Fracción de Greenwald máxima


@dataclass 
class DisruptionParameters:
    """
    Parámetros que controlan la física de la disrupción simulada.
    
    Física de los precursores:
    --------------------------
    Los precursores son señales de advertencia que aparecen antes de una disrupción.
    Típicamente incluyen:
    
    1. Oscilaciones MHD crecientes: Modos tearing/locked modes que crecen
       exponencialmente. Detectables en señales de Mirnov y en fluctuaciones
       de temperatura/densidad.
    
    2. Aproximación a límites operacionales:
       - βN aumenta hacia el límite de Troyon
       - q95 disminuye hacia el valor crítico
       - li aumenta (perfil de corriente se contrae)
    
    Escalas temporales típicas:
    ---------------------------
    - Precursores: 10-100 ms antes del colapso
    - Thermal quench: 1-3 ms (pérdida de energía térmica)
    - Current quench: 10-100 ms (decaimiento de corriente)
    """
    # Duración de fases (como fracción del tiempo total de ventana)
    precursor_start: float = 0.3    # Los precursores empiezan al 30% del tiempo
    thermal_quench_start: float = 0.85  # El colapso térmico empieza al 85%
    
    # Intensidad de precursores MHD
    mhd_oscillation_freq: float = 50.0  # Frecuencia base de oscilaciones [Hz normalizado]
    mhd_growth_rate: float = 3.0        # Tasa de crecimiento exponencial
    mhd_amplitude: float = 0.1          # Amplitud inicial de oscilaciones
    
    # Cambios en parámetros durante precursores
    betan_increase: float = 0.8     # Cuánto aumenta βN hacia el límite
    q95_decrease: float = 1.5       # Cuánto disminuye q95 hacia valor crítico
    li_increase: float = 0.3        # Cuánto aumenta li (contracción del perfil)
    density_spike: float = 0.5      # Posible aumento de densidad
    
    # Parámetros del colapso (thermal + current quench)
    collapse_rate: float = 20.0     # Qué tan rápido colapsa Ip
    thermal_quench_amplitude: float = 0.8  # Fracción de energía perdida


class SyntheticTokamakData:
    """
    Generador de datos sintéticos de tokamak para predicción de disrupciones.
    
    Esta clase genera disparos (shots) sintéticos que simulan las señales
    diagnósticas de un tokamak real. Los disparos pueden ser:
    
    - Normales: Operación estable sin disrupción
    - Disruptivos: Muestran precursores seguidos de colapso
    
    Uso típico:
    -----------
    ```python
    generator = SyntheticTokamakData(seed=42)
    dataset = generator.generate_dataset(n_normal=500, n_disruptive=500)
    generator.save_to_hdf5(dataset, 'data/tokamak_synthetic.h5')
    ```
    
    Física simulada:
    ----------------
    1. Señales base con fluctuaciones turbulentas (microinestabilidades ITG/TEM)
    2. Ruido instrumental gaussiano (~2-5%)
    3. Precursores MHD (oscilaciones que crecen exponencialmente)
    4. Aproximación a límites operacionales antes de disrupción
    5. Colapso abrupto (thermal quench + current quench)
    """
    
    def __init__(
        self,
        tokamak_params: Optional[TokamakParameters] = None,
        disruption_params: Optional[DisruptionParameters] = None,
        sampling_rate: int = 10000,  # 10 kHz típico de diagnósticos
        shot_duration: float = 0.1,   # 100 ms de ventana temporal
        seed: Optional[int] = None
    ):
        """
        Inicializa el generador de datos sintéticos.
        
        Args:
            tokamak_params: Parámetros del tokamak (usa DIII-D por defecto)
            disruption_params: Parámetros de la física de disrupciones
            sampling_rate: Frecuencia de muestreo en Hz (10 kHz = típico)
            shot_duration: Duración de la ventana temporal en segundos
            seed: Semilla para reproducibilidad de resultados
        
        Física del muestreo:
        --------------------
        - 10 kHz es la frecuencia típica de diagnósticos ECE, interferometría
        - 100 ms de ventana captura los precursores típicos (10-100 ms antes)
        - Con estos valores: 1000 puntos por disparo
        """
        self.tokamak = tokamak_params or TokamakParameters()
        self.disruption = disruption_params or DisruptionParameters()
        self.sampling_rate = sampling_rate
        self.shot_duration = shot_duration
        self.n_samples = int(sampling_rate * shot_duration)
        
        # Configurar semilla para reproducibilidad
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        # Vector de tiempo normalizado [0, 1]
        self.t = np.linspace(0, 1, self.n_samples)
        
        # Nombres de las señales (canales)
        self.signal_names = ['ip', 'betan', 'q95', 'density', 'li']
    
    def _generate_base_signal(
        self,
        mean: float,
        std: float,
        drift: float = 0.0,
        noise_level: float = 0.02
    ) -> np.ndarray:
        """
        Genera una señal base estable con ruido instrumental.
        
        Física del plasma turbulento:
        -----------------------------
        Las señales de un tokamak nunca son perfectamente constantes debido a:
        
        1. Turbulencia del plasma: Microinestabilidades (ITG, TEM, ETG) causan
           fluctuaciones del 1-5% en temperatura y densidad. Estas tienen
           espectro de frecuencias amplio pero dominan las bajas frecuencias.
        
        2. Ruido instrumental: Los diagnósticos tienen ruido electrónico
           gaussiano. ECE típicamente ~2-3%, interferometría ~1%.
        
        3. Drift temporal: Durante un disparo, los perfiles pueden evolucionar
           lentamente (calentamiento progresivo, acumulación de impurezas).
        
        Args:
            mean: Valor medio de la señal
            std: Desviación estándar del valor base (variabilidad entre disparos)
            drift: Pendiente de drift temporal (cambio lento durante el disparo)
            noise_level: Nivel de ruido como fracción del valor medio
        
        Returns:
            Array de longitud n_samples con la señal temporal
        """
        # Valor base para este disparo específico (varía entre disparos)
        base_value = np.random.normal(mean, std)
        base_value = max(base_value, mean * 0.5)  # Evitar valores negativos/muy bajos
        
        # Evolución temporal suave (drift lineal)
        signal = base_value * (1 + drift * self.t)
        
        # Añadir fluctuaciones de baja frecuencia (simula turbulencia de plasma)
        # Estas son las fluctuaciones "reales" del plasma, no ruido instrumental
        n_modes = 5
        for i in range(1, n_modes + 1):
            amplitude = noise_level * base_value / (i * 2)  # Espectro 1/f
            frequency = i * 2 * np.pi  # Frecuencias bajas
            phase = np.random.uniform(0, 2 * np.pi)
            signal += amplitude * np.sin(frequency * self.t + phase)
        
        # Ruido gaussiano de alta frecuencia (ruido instrumental)
        noise = np.random.normal(0, noise_level * base_value, self.n_samples)
        signal += noise
        
        return signal
    
    def _add_mhd_oscillations(
        self,
        signal: np.ndarray,
        start_time: float,
        amplitude: float,
        growth_rate: float,
        frequency: float
    ) -> np.ndarray:
        """
        Añade oscilaciones MHD crecientes a una señal (precursor de disrupción).
        
        Física de las inestabilidades MHD:
        ----------------------------------
        Los modos MHD (tearing modes, locked modes) son perturbaciones del
        campo magnético que crecen cuando el plasma se acerca a límites de
        estabilidad. Características:
        
        1. Crecimiento exponencial: La amplitud crece como exp(γt) donde γ
           es la tasa de crecimiento. Típicamente γ ~ 10-100 s⁻¹.
        
        2. Frecuencia de rotación: Los modos rotan con el plasma a ~1-30 kHz.
           Cuando se "bloquean" (locked mode), la frecuencia cae a cero y
           la disrupción es inminente.
        
        3. Estructura espacial: Modos (m,n) donde m es número poloidal y n
           toroidal. El modo (2,1) en la superficie q=2 es el más peligroso.
        
        Args:
            signal: Señal base a modificar
            start_time: Tiempo normalizado [0,1] donde empiezan las oscilaciones
            amplitude: Amplitud inicial de las oscilaciones
            growth_rate: Tasa de crecimiento exponencial
            frequency: Frecuencia de oscilación
        
        Returns:
            Señal modificada con oscilaciones MHD
        """
        modified_signal = signal.copy()
        
        # Máscara para la región donde hay oscilaciones
        mask = self.t >= start_time
        t_shifted = self.t[mask] - start_time
        
        # Crecimiento exponencial de la amplitud
        # Física: dA/dt = γA → A(t) = A₀ exp(γt)
        envelope = amplitude * np.exp(growth_rate * t_shifted)
        
        # Oscilación sinusoidal (modo MHD rotando)
        # Añadimos múltiples armónicos para mayor realismo
        oscillation = np.zeros_like(t_shifted)
        for harmonic in [1, 2, 3]:
            phase = np.random.uniform(0, 2 * np.pi)
            oscillation += np.sin(2 * np.pi * frequency * harmonic * t_shifted + phase) / harmonic
        
        # Aplicar oscilaciones a la señal
        modified_signal[mask] += envelope * oscillation * np.mean(signal)
        
        return modified_signal
    
    def _apply_thermal_quench(
        self,
        signal: np.ndarray,
        start_time: float,
        collapse_rate: float,
        final_fraction: float
    ) -> np.ndarray:
        """
        Aplica el colapso térmico (thermal quench) a una señal.
        
        Física del Thermal Quench:
        --------------------------
        El thermal quench es la pérdida rápida de energía térmica del plasma.
        Ocurre cuando las islas magnéticas se solapan y crean regiones de
        campo estocástico. El calor escapa siguiendo las líneas de campo
        caóticas hacia las paredes.
        
        Características:
        - Duración: 1-3 ms en tokamaks actuales, ~70-100 ms proyectado para ITER
        - La temperatura cae de keV a ~10 eV (2-3 órdenes de magnitud)
        - La energía se deposita en los componentes de cara al plasma
        
        Modelamos el colapso como una función sigmoide invertida (tanh):
        
        Args:
            signal: Señal a modificar
            start_time: Tiempo normalizado donde empieza el colapso
            collapse_rate: Qué tan abrupto es el colapso
            final_fraction: Fracción del valor original que queda después
        
        Returns:
            Señal con thermal quench aplicado
        """
        modified_signal = signal.copy()
        
        # Función de colapso suave usando tanh
        # tanh va de -1 a 1, lo transformamos a ir de 1 a final_fraction
        collapse_profile = 0.5 * (1 - np.tanh(collapse_rate * (self.t - start_time)))
        collapse_profile = final_fraction + (1 - final_fraction) * collapse_profile
        
        modified_signal *= collapse_profile
        
        return modified_signal
    
    def generate_normal_shot(self) -> Dict[str, np.ndarray]:
        """
        Genera un disparo sin disrupción (operación normal).
        
        Física de operación normal:
        ---------------------------
        En un disparo normal, todas las señales se mantienen dentro de los
        límites operacionales seguros:
        
        - Ip: Controlada activamente por el sistema de control (muy estable)
        - βN: ~50-70% del límite de Troyon (margen de seguridad)
        - q95: > 3 (muy por encima del límite crítico de 2)
        - Densidad: < 80% del límite de Greenwald
        - li: Estable (perfil de corriente no se contrae)
        
        Las fluctuaciones son pequeñas y corresponden a turbulencia normal
        del plasma y ruido instrumental.
        
        Returns:
            Diccionario con las 5 señales diagnósticas, cada una de longitud n_samples
        """
        shot = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # CORRIENTE DE PLASMA (Ip) [MA]
        # ═══════════════════════════════════════════════════════════════════
        # Física: La corriente de plasma genera el campo poloidal que, junto
        # con el campo toroidal, crea las superficies magnéticas cerradas.
        # El sistema de control ajusta el voltaje del transformador central
        # para mantener Ip constante → señal muy estable.
        shot['ip'] = self._generate_base_signal(
            mean=np.random.uniform(*self.tokamak.Ip_range),
            std=0.05,       # Poca variabilidad entre disparos
            drift=0.0,      # Control activo mantiene Ip constante
            noise_level=0.01  # Señal muy limpia
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # BETA NORMALIZADO (βN)
        # ═══════════════════════════════════════════════════════════════════
        # Física: βN = β × a × B / Ip donde β = 2μ₀<p>/B²
        # Mide la eficiencia del confinamiento: cuánta presión de plasma
        # logramos confinar por unidad de presión magnética.
        # En operación normal, se mantiene ~50-70% del límite de Troyon (~3.5)
        shot['betan'] = self._generate_base_signal(
            mean=np.random.uniform(*self.tokamak.betan_range),
            std=0.2,
            drift=0.02,     # Puede aumentar ligeramente con calentamiento
            noise_level=0.03
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # FACTOR DE SEGURIDAD (q95)
        # ═══════════════════════════════════════════════════════════════════
        # Física: q = (r × Bt) / (R × Bp) indica cuántas vueltas toroidales
        # da una línea de campo por cada vuelta poloidal.
        # q95 se evalúa en la superficie que encierra 95% del flujo poloidal.
        # q95 < 2 → inestabilidad kink externa → disrupción casi segura
        # Operación normal: q95 ~ 3-5
        shot['q95'] = self._generate_base_signal(
            mean=np.random.uniform(*self.tokamak.q95_range),
            std=0.3,
            drift=-0.01,    # Puede bajar ligeramente si Ip aumenta
            noise_level=0.02
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # DENSIDAD ELECTRÓNICA (ne) [10^19 m^-3]
        # ═══════════════════════════════════════════════════════════════════
        # Física: La densidad se controla mediante inyección de gas y pellets.
        # Límite de Greenwald: nG = Ip / (π a²) en unidades de 10²⁰ m⁻³
        # Operar cerca del límite → alta radiación → enfriamiento → disrupción
        shot['density'] = self._generate_base_signal(
            mean=np.random.uniform(*self.tokamak.density_range),
            std=0.5,
            drift=0.01,     # Puede aumentar con fueling continuo
            noise_level=0.02
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # INDUCTANCIA INTERNA (li)
        # ═══════════════════════════════════════════════════════════════════
        # Física: li indica qué tan "picudo" es el perfil de corriente.
        # li alto = corriente concentrada en el centro (perfil estrecho)
        # li bajo = corriente distribuida (perfil ancho, más estable)
        # Cuando el borde se enfría, la corriente se contrae → li aumenta
        # Esto es un precursor de problemas.
        shot['li'] = self._generate_base_signal(
            mean=np.random.uniform(*self.tokamak.li_range),
            std=0.05,
            drift=0.0,
            noise_level=0.02
        )
        
        return shot
    
    def generate_disruptive_shot(self) -> Dict[str, np.ndarray]:
        """
        Genera un disparo con disrupción (precursores + colapso).
        
        Física de una disrupción:
        -------------------------
        Una disrupción ocurre cuando el plasma pierde el confinamiento de
        manera abrupta. La secuencia típica es:
        
        1. FASE DE PRECURSORES (10-100 ms antes):
           - Oscilaciones MHD crecientes (modos tearing/locked modes)
           - βN aumenta hacia el límite de Troyon
           - q95 disminuye hacia el valor crítico (~2)
           - li aumenta (el perfil de corriente se contrae)
           - Posible aumento de densidad/radiación
        
        2. THERMAL QUENCH (1-3 ms):
           - Pérdida rápida de energía térmica
           - Campo magnético se vuelve estocástico
           - Temperatura cae de keV a ~10 eV
        
        3. CURRENT QUENCH (10-100 ms):
           - Decaimiento de la corriente de plasma
           - La resistividad aumenta dramáticamente con T baja
           - Generación potencial de electrones desbocados
        
        Returns:
            Diccionario con las 5 señales mostrando la secuencia disruptiva
        """
        # Empezar con un disparo "normal" como base
        shot = self.generate_normal_shot()
        
        # Parámetros de tiempo para las fases
        precursor_start = self.disruption.precursor_start
        thermal_quench_start = self.disruption.thermal_quench_start
        
        # ═══════════════════════════════════════════════════════════════════
        # FASE DE PRECURSORES: Modificar cada señal
        # ═══════════════════════════════════════════════════════════════════
        
        # --- Corriente de plasma (Ip) ---
        # Durante precursores: relativamente estable pero con oscilaciones MHD
        # Durante thermal quench: colapso abrupto
        shot['ip'] = self._add_mhd_oscillations(
            shot['ip'],
            start_time=precursor_start,
            amplitude=self.disruption.mhd_amplitude * 0.5,  # Menos visible en Ip
            growth_rate=self.disruption.mhd_growth_rate,
            frequency=self.disruption.mhd_oscillation_freq
        )
        shot['ip'] = self._apply_thermal_quench(
            shot['ip'],
            start_time=thermal_quench_start,
            collapse_rate=self.disruption.collapse_rate,
            final_fraction=0.1  # Ip cae al 10% de su valor
        )
        
        # --- Beta normalizado (βN) ---
        # Física: Aumenta hacia el límite de Troyon antes de la disrupción
        # Esto puede ser causa o consecuencia de la inestabilidad
        betan_base = shot['betan'].copy()
        
        # Rampa hacia el límite de Troyon durante precursores
        mask = self.t >= precursor_start
        t_precursor = (self.t[mask] - precursor_start) / (thermal_quench_start - precursor_start)
        betan_increase = self.disruption.betan_increase * t_precursor
        shot['betan'][mask] = betan_base[mask] + betan_increase * np.mean(betan_base)
        
        # Añadir oscilaciones MHD
        shot['betan'] = self._add_mhd_oscillations(
            shot['betan'],
            start_time=precursor_start,
            amplitude=self.disruption.mhd_amplitude,
            growth_rate=self.disruption.mhd_growth_rate,
            frequency=self.disruption.mhd_oscillation_freq
        )
        
        # Colapso durante thermal quench
        shot['betan'] = self._apply_thermal_quench(
            shot['betan'],
            start_time=thermal_quench_start,
            collapse_rate=self.disruption.collapse_rate * 1.5,  # Más rápido que Ip
            final_fraction=0.05
        )
        
        # --- Factor de seguridad (q95) ---
        # Física: Disminuye hacia el valor crítico (~2)
        # q95 = (a² × B) / (R × μ₀ × Ip) → si Ip sube sin aumentar B, q baja
        q95_base = shot['q95'].copy()
        
        # Rampa hacia abajo durante precursores
        mask = self.t >= precursor_start
        t_precursor = (self.t[mask] - precursor_start) / (thermal_quench_start - precursor_start)
        q95_decrease = self.disruption.q95_decrease * t_precursor
        shot['q95'][mask] = q95_base[mask] - q95_decrease
        
        # Asegurar que no baje de 1.5 (físicamente poco realista)
        shot['q95'] = np.maximum(shot['q95'], 1.5)
        
        # Oscilaciones MHD en q95
        shot['q95'] = self._add_mhd_oscillations(
            shot['q95'],
            start_time=precursor_start,
            amplitude=self.disruption.mhd_amplitude * 0.7,
            growth_rate=self.disruption.mhd_growth_rate,
            frequency=self.disruption.mhd_oscillation_freq
        )
        
        # --- Densidad electrónica ---
        # Física: Puede aumentar (acumulación de impurezas, MARFE)
        # o mantenerse estable dependiendo del tipo de disrupción
        density_base = shot['density'].copy()
        
        # Posible spike de densidad (simula MARFE o inyección de impurezas)
        if np.random.random() > 0.5:  # 50% de probabilidad de spike
            mask = self.t >= precursor_start
            t_precursor = (self.t[mask] - precursor_start) / (thermal_quench_start - precursor_start)
            density_increase = self.disruption.density_spike * t_precursor**2
            shot['density'][mask] = density_base[mask] * (1 + density_increase)
        
        # Oscilaciones
        shot['density'] = self._add_mhd_oscillations(
            shot['density'],
            start_time=precursor_start,
            amplitude=self.disruption.mhd_amplitude * 0.8,
            growth_rate=self.disruption.mhd_growth_rate,
            frequency=self.disruption.mhd_oscillation_freq
        )
        
        # Colapso (la densidad puede aumentar brevemente durante el quench
        # por ablación de paredes, pero luego cae)
        shot['density'] = self._apply_thermal_quench(
            shot['density'],
            start_time=thermal_quench_start + 0.02,  # Ligeramente retrasado
            collapse_rate=self.disruption.collapse_rate * 0.8,
            final_fraction=0.2
        )
        
        # --- Inductancia interna (li) ---
        # Física: AUMENTA antes de la disrupción
        # Cuando el borde se enfría (radiación, pérdida de confinamiento),
        # la resistividad del borde aumenta y la corriente se contrae
        # hacia el centro → perfil más picudo → li aumenta
        li_base = shot['li'].copy()
        
        # Rampa hacia arriba durante precursores
        mask = self.t >= precursor_start
        t_precursor = (self.t[mask] - precursor_start) / (thermal_quench_start - precursor_start)
        li_increase = self.disruption.li_increase * t_precursor**1.5
        shot['li'][mask] = li_base[mask] + li_increase
        
        # Oscilaciones más sutiles en li
        shot['li'] = self._add_mhd_oscillations(
            shot['li'],
            start_time=precursor_start,
            amplitude=self.disruption.mhd_amplitude * 0.5,
            growth_rate=self.disruption.mhd_growth_rate * 0.8,
            frequency=self.disruption.mhd_oscillation_freq
        )
        
        return shot
    
    def generate_dataset(
        self,
        n_normal: int = 500,
        n_disruptive: int = 500,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Genera un dataset completo con disparos normales y disruptivos.
        
        Args:
            n_normal: Número de disparos normales (label = 0)
            n_disruptive: Número de disparos disruptivos (label = 1)
            verbose: Si True, muestra barra de progreso
        
        Returns:
            Diccionario con:
            - 'data': Array de forma (n_total, n_channels, n_samples)
            - 'labels': Array de forma (n_total,) con 0=normal, 1=disruptivo
            - 'signal_names': Lista de nombres de canales
        """
        n_total = n_normal + n_disruptive
        n_channels = len(self.signal_names)
        
        # Preallocar arrays
        data = np.zeros((n_total, n_channels, self.n_samples), dtype=np.float32)
        labels = np.zeros(n_total, dtype=np.int64)
        
        # Generar disparos normales
        iterator = range(n_normal)
        if verbose:
            iterator = tqdm(iterator, desc="Generando disparos normales")
        
        for i in iterator:
            shot = self.generate_normal_shot()
            for j, name in enumerate(self.signal_names):
                data[i, j, :] = shot[name]
            labels[i] = 0
        
        # Generar disparos disruptivos
        iterator = range(n_disruptive)
        if verbose:
            iterator = tqdm(iterator, desc="Generando disparos disruptivos")
        
        for i in iterator:
            shot = self.generate_disruptive_shot()
            for j, name in enumerate(self.signal_names):
                data[n_normal + i, j, :] = shot[name]
            labels[n_normal + i] = 1
        
        # Mezclar el dataset
        if self.seed is not None:
            np.random.seed(self.seed + 1000)  # Semilla diferente para shuffle
        
        shuffle_idx = np.random.permutation(n_total)
        data = data[shuffle_idx]
        labels = labels[shuffle_idx]
        
        return {
            'data': data,
            'labels': labels,
            'signal_names': self.signal_names,
            'n_samples': self.n_samples,
            'sampling_rate': self.sampling_rate,
            'shot_duration': self.shot_duration
        }
    
    def save_to_hdf5(self, dataset: Dict, filepath: str) -> None:
        """
        Guarda el dataset en formato HDF5.
        
        HDF5 es el formato estándar para datos científicos grandes.
        Ventajas:
        - Compresión eficiente
        - Acceso aleatorio a subconjuntos
        - Metadatos integrados
        - Compatible con h5py, PyTorch, TensorFlow
        
        Args:
            dataset: Diccionario retornado por generate_dataset()
            filepath: Ruta donde guardar el archivo
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Calcular estadísticas antes de guardar
        n_normal = int(np.sum(dataset['labels'] == 0))
        n_disruptive = int(np.sum(dataset['labels'] == 1))
        
        with h5py.File(filepath, 'w') as f:
            # Datos principales
            f.create_dataset('data', data=dataset['data'], compression='gzip')
            f.create_dataset('labels', data=dataset['labels'])
            
            # Metadatos
            f.attrs['signal_names'] = dataset['signal_names']
            f.attrs['n_samples'] = dataset['n_samples']
            f.attrs['sampling_rate'] = dataset['sampling_rate']
            f.attrs['shot_duration'] = dataset['shot_duration']
            f.attrs['n_normal'] = n_normal
            f.attrs['n_disruptive'] = n_disruptive
            
            # Información del generador
            f.attrs['generator'] = 'SyntheticTokamakData'
            f.attrs['seed'] = self.seed if self.seed is not None else -1
        
        print(f"Dataset guardado en: {filepath}")
        print(f"  - Disparos normales: {n_normal}")
        print(f"  - Disparos disruptivos: {n_disruptive}")
        print(f"  - Shape de datos: {dataset['data'].shape}")
    
    @staticmethod
    def load_from_hdf5(filepath: str) -> Dict:
        """
        Carga un dataset desde archivo HDF5.
        
        Args:
            filepath: Ruta del archivo HDF5
        
        Returns:
            Diccionario con datos y metadatos
        """
        with h5py.File(filepath, 'r') as f:
            dataset = {
                'data': f['data'][:],
                'labels': f['labels'][:],
                'signal_names': list(f.attrs['signal_names']),
                'n_samples': f.attrs['n_samples'],
                'sampling_rate': f.attrs['sampling_rate'],
                'shot_duration': f.attrs['shot_duration']
            }
        
        print(f"Dataset cargado desde: {filepath}")
        print(f"  - Shape: {dataset['data'].shape}")
        print(f"  - Clases: {np.bincount(dataset['labels'])}")
        
        return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE UTILIDAD
# ═══════════════════════════════════════════════════════════════════════════════

def create_default_dataset(
    output_path: str = 'data/tokamak_synthetic.h5',
    n_normal: int = 500,
    n_disruptive: int = 500,
    seed: int = 42
) -> Dict:
    """
    Función de conveniencia para crear el dataset por defecto.
    
    Args:
        output_path: Ruta donde guardar el archivo HDF5
        n_normal: Número de disparos normales
        n_disruptive: Número de disparos disruptivos
        seed: Semilla para reproducibilidad
    
    Returns:
        Dataset generado
    """
    print("=" * 60)
    print("GENERADOR DE DATOS SINTÉTICOS DE TOKAMAK")
    print("=" * 60)
    print(f"\nParámetros:")
    print(f"  - Disparos normales: {n_normal}")
    print(f"  - Disparos disruptivos: {n_disruptive}")
    print(f"  - Semilla: {seed}")
    print(f"  - Archivo de salida: {output_path}")
    print()
    
    generator = SyntheticTokamakData(seed=seed)
    dataset = generator.generate_dataset(n_normal=n_normal, n_disruptive=n_disruptive)
    generator.save_to_hdf5(dataset, output_path)
    
    print("\n" + "=" * 60)
    print("¡Dataset generado exitosamente!")
    print("=" * 60)
    
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN DIRECTA
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Crear dataset por defecto cuando se ejecuta directamente
    dataset = create_default_dataset()
    
    # Mostrar información adicional
    print("\nInformación del dataset:")
    print(f"  - Forma de datos: {dataset['data'].shape}")
    print(f"    → {dataset['data'].shape[0]} disparos")
    print(f"    → {dataset['data'].shape[1]} canales (señales)")
    print(f"    → {dataset['data'].shape[2]} muestras temporales")
    print(f"  - Canales: {dataset['signal_names']}")
    print(f"  - Frecuencia de muestreo: {dataset['sampling_rate']} Hz")
    print(f"  - Duración de ventana: {dataset['shot_duration']*1000} ms")
