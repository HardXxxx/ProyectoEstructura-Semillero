"""
test_cold_chain.py — Pruebas unitarias para la simulación de cadena de frío.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.nodo import Nodo
from src.models.arista import Arista
from src.models.grafo_logistico import GrafoLogistico
from src.optimization.cold_chain import SimuladorCadenaFrio


def _crear_grafo_frio() -> GrafoLogistico:
    """Grafo de prueba con y sin centros de frío."""
    g = GrafoLogistico()

    g.agregar_nodo(Nodo("P01", "Pesca", "P", produccion_kg_dia=100))
    g.agregar_nodo(Nodo("F01", "Frío", "F", capacidad_frio_kg=5000))
    g.agregar_nodo(Nodo("A01", "Acopio", "A", capacidad_frio_kg=0))
    g.agregar_nodo(Nodo("V01", "Mercado", "V"))

    g.agregar_arista(Arista("P01", "F01", distancia_km=10, tiempo_h=0.5))
    g.agregar_arista(Arista("P01", "A01", distancia_km=15, tiempo_h=1.5))
    g.agregar_arista(Arista("F01", "V01", distancia_km=200, tiempo_h=4.0))
    g.agregar_arista(Arista("A01", "V01", distancia_km=210, tiempo_h=5.0))

    return g


def test_simulacion_con_frio():
    """Simula ruta pasando por centro de frío."""
    g = _crear_grafo_frio()
    sim = SimuladorCadenaFrio(g, tiempo_limite_h=4.0)
    resultado = sim.simular_ruta(["P01", "F01", "V01"], 100.0)

    assert resultado.kg_inicial == 100.0
    assert resultado.kg_final > 0
    assert resultado.kg_final <= 100.0
    assert resultado.num_paradas_frio >= 1
    assert resultado.tiempo_total_h > 0
    print(f"  [✓] test_simulacion_con_frio: {resultado.kg_final:.1f} kg final, "
          f"{resultado.num_paradas_frio} parada(s) frío")


def test_simulacion_sin_frio():
    """Simula ruta sin ningún centro de frío."""
    g = _crear_grafo_frio()
    sim = SimuladorCadenaFrio(g, tiempo_limite_h=4.0)
    resultado = sim.simular_ruta(["P01", "A01", "V01"], 100.0)

    assert resultado.num_paradas_frio == 0
    assert resultado.perdida_kg > 0  # Hay deterioro sin frío
    assert resultado.cumple_limite is False  # 1.5 + 5.0 = 6.5h > 4h
    print(f"  [✓] test_simulacion_sin_frio: pérdida={resultado.perdida_pct:.1f}%, "
          f"cumple={resultado.cumple_limite}")


def test_simulacion_deterioro_mayor_sin_frio():
    """Verifica que sin frío el deterioro es mayor que con frío."""
    g = _crear_grafo_frio()
    sim = SimuladorCadenaFrio(g, tiempo_limite_h=4.0)

    con_frio = sim.simular_ruta(["P01", "F01", "V01"], 100.0)
    sin_frio = sim.simular_ruta(["P01", "A01", "V01"], 100.0)

    # Sin frío debería tener más pérdida (ruta más larga y sin refrigeración)
    assert sin_frio.perdida_pct >= con_frio.perdida_pct
    print(f"  [✓] test_simulacion_deterioro_mayor_sin_frio: "
          f"con_frio={con_frio.perdida_pct:.1f}% vs sin_frio={sin_frio.perdida_pct:.1f}%")


def test_simulacion_ruta_corta():
    """Verifica que una ruta corta cumple el límite térmico."""
    g = _crear_grafo_frio()
    sim = SimuladorCadenaFrio(g, tiempo_limite_h=4.0)
    resultado = sim.simular_ruta(["P01", "F01"], 50.0)

    assert resultado.tiempo_total_h == 0.5
    assert resultado.kg_final > 0
    # F01 tiene frío, el timer se resetea → tiempo_sin_frio = 0
    assert resultado.cumple_limite is True
    print(f"  [✓] test_simulacion_ruta_corta: {resultado.tiempo_total_h:.2f}h, "
          f"cumple={resultado.cumple_limite}")


def test_simulacion_resumen():
    """Verifica que el resumen se genera sin errores."""
    g = _crear_grafo_frio()
    sim = SimuladorCadenaFrio(g, tiempo_limite_h=4.0)
    resultado = sim.simular_ruta(["P01", "F01", "V01"], 80.0)
    texto = resultado.resumen()
    assert "SIMULACIÓN" in texto
    assert "Kg" in texto or "kg" in texto.lower()
    print("  [✓] test_simulacion_resumen")


if __name__ == "__main__":
    print("\n═══ PRUEBAS DE CADENA DE FRÍO ═══\n")
    test_simulacion_con_frio()
    test_simulacion_sin_frio()
    test_simulacion_deterioro_mayor_sin_frio()
    test_simulacion_ruta_corta()
    test_simulacion_resumen()
    print("\n  ✅ Todas las pruebas pasaron.\n")
