"""
scripts/test_fno.py

Script de verificación del FNO.
Verifica: forward pass, backward pass, shapes, parámetros.
"""

import sys
sys.path.insert(0, '.')

import torch
from src.models.fno import SpectralConv1d, FourierLayer, FNO1d, count_fno_parameters


def test_spectral_conv():
    """Test SpectralConv1d"""
    print("=" * 50)
    print("Testing SpectralConv1d")
    print("=" * 50)
    
    # Configuración
    batch_size = 4
    in_channels = 5
    out_channels = 32
    time_steps = 1000
    modes = 16
    
    # Crear capa
    layer = SpectralConv1d(in_channels, out_channels, modes)
    
    # Input aleatorio
    x = torch.randn(batch_size, in_channels, time_steps)
    
    # Forward pass
    y = layer(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected:     [{batch_size}, {out_channels}, {time_steps}]")
    
    # Verificar shape
    assert y.shape == (batch_size, out_channels, time_steps), "Shape mismatch!"
    print("✓ Forward pass OK")
    
    # Test backward
    loss = y.sum()
    loss.backward()
    print("✓ Backward pass OK")
    
    # Contar parámetros
    n_params = sum(p.numel() for p in layer.parameters())
    print(f"✓ Parameters: {n_params:,} (complex weights)")
    
    return True


def test_fourier_layer():
    """Test FourierLayer"""
    print("\n" + "=" * 50)
    print("Testing FourierLayer")
    print("=" * 50)
    
    batch_size = 4
    channels = 32
    time_steps = 1000
    modes = 16
    
    layer = FourierLayer(channels, modes)
    x = torch.randn(batch_size, channels, time_steps)
    
    y = layer(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    
    assert y.shape == x.shape, "Shape mismatch!"
    print("✓ Forward pass OK (shape preserved)")
    
    loss = y.sum()
    loss.backward()
    print("✓ Backward pass OK")
    
    return True


def test_fno1d():
    """Test FNO1d completo"""
    print("\n" + "=" * 50)
    print("Testing FNO1d (Full Model)")
    print("=" * 50)
    
    batch_size = 4
    in_channels = 5
    out_channels = 2
    time_steps = 1000
    
    # Crear modelo con configuración del plan
    model = FNO1d(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=32,
        modes=16,
        n_layers=3
    )
    
    # Input
    x = torch.randn(batch_size, in_channels, time_steps)
    
    # Forward
    y = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected:     [{batch_size}, {out_channels}]")
    
    assert y.shape == (batch_size, out_channels), "Shape mismatch!"
    print("✓ Forward pass OK")
    
    # Backward
    loss = y.sum()
    loss.backward()
    print("✓ Backward pass OK")
    
    # Parámetros
    params = count_fno_parameters(model)
    print(f"✓ Parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    
    # Verificar que está bajo 500k (objetivo del plan)
    if params['total'] < 500_000:
        print(f"✓ Under 500k parameter limit!")
    else:
        print(f"⚠ Over 500k parameter limit")
    
    return True


def test_different_resolutions():
    """Test que FNO funciona con diferentes resoluciones temporales"""
    print("\n" + "=" * 50)
    print("Testing Resolution Invariance")
    print("=" * 50)
    
    model = FNO1d(in_channels=5, out_channels=2, hidden_channels=32, modes=16, n_layers=3)
    model.eval()  # Modo evaluación
    
    resolutions = [500, 1000, 2000]
    
    for res in resolutions:
        x = torch.randn(2, 5, res)
        y = model(x)
        print(f"Resolution {res:4d}: Input {x.shape} → Output {y.shape}")
        assert y.shape == (2, 2), f"Failed at resolution {res}"
    
    print("✓ Resolution invariance OK!")
    return True


if __name__ == "__main__":
    print("\n🔬 FNO Verification Tests\n")
    
    all_passed = True
    
    try:
        test_spectral_conv()
        test_fourier_layer()
        test_fno1d()
        test_different_resolutions()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        all_passed = False
    
    if all_passed:
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
        print("\nEl FNO está listo para entrenamiento (Día 9)")