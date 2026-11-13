"""
arista.py — Modelo de datos para las aristas del grafo logístico.

Cada arista conecta dos nodos y almacena atributos logísticos
multicriterio: distancia, tiempo, costo, riesgo térmico y
pérdida esperada. El peso combinado se calcula con la función
multicriterio configurable.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from ..utils.constants import (
    ALPHA, BETA, GAMMA, DELTA,
    COSTO_TRANSPORTE_KM,
    TASA_DETERIORO_BASE,
)


@dataclass
class Arista:
    """
    Representa una conexión dirigida entre dos nodos del grafo.

    Attributes
    ----------
    origen : str
        ID del nodo de origen.
    destino : str
        ID del nodo de destino.
    distancia_km : float
        Distancia del tramo en kilómetros.
    tiempo_h : float
        Tiempo de recorrido estimado en horas.
    costo_cop : float
        Costo de transporte del tramo (COP).
    riesgo_termico : float
        Índice de riesgo de ruptura de cadena de frío (0.0 – 1.0).
    estado_via : str
        Estado de la vía: 'buena', 'regular', 'mala'.
    tipo_via : str
        Tipo de conexión: 'terrestre', 'fluvial', 'acceso_local'.
    capacidad_kg : float
        Capacidad máxima de transporte en el tramo.
    num_transbordos : int
        Número de cambios de vehículo/contenedor en el tramo.
    metadata : dict
        Atributos adicionales flexibles.
    """

    origen: str
    destino: str
    distancia_km: float = 0.0
    tiempo_h: float = 0.0
    costo_cop: float = 0.0
    riesgo_termico: float = 0.0
    estado_via: str = "buena"
    tipo_via: str = "terrestre"
    capacidad_kg: float = 2000.0
    num_transbordos: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calcula costo por defecto si no fue provisto."""
        if self.costo_cop == 0.0 and self.distancia_km > 0:
            self.costo_cop = self.distancia_km * COSTO_TRANSPORTE_KM

    # ── Peso multicriterio ──────────────────────────────────────

    @property
    def factor_estado_via(self) -> float:
        """Multiplicador de penalización por estado de la vía."""
        factores = {"buena": 1.0, "regular": 1.5, "mala": 2.0}
        return factores.get(self.estado_via, 1.0)

    @property
    def perdida_esperada(self) -> float:
        """Porcentaje de pérdida estimada en el tramo."""
        return min(
            TASA_DETERIORO_BASE * self.tiempo_h * self.factor_estado_via,
            100.0
        )

    def peso(self, alpha: float = ALPHA, beta: float = BETA,
             gamma: float = GAMMA, delta: float = DELTA) -> float:
        """
        Calcula el peso combinado de la arista.

        w_ij = α·tiempo + β·costo_normalizado + γ·riesgo + δ·pérdida

        Los valores se normalizan para que sean comparables:
        - tiempo:  horas (0–48)
        - costo:   millones de COP (0–5)
        - riesgo:  0.0–1.0
        - pérdida: 0.0–1.0
        """
        t_norm = self.tiempo_h / 48.0
        c_norm = self.costo_cop / 5_000_000.0
        r_norm = self.riesgo_termico
        p_norm = self.perdida_esperada / 100.0

        return (alpha * t_norm
                + beta * c_norm
                + gamma * r_norm
                + delta * p_norm)

    @property
    def peso_default(self) -> float:
        """Peso con coeficientes por defecto."""
        return self.peso()

    # ── Serialización ───────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la arista a diccionario serializable."""
        return {
            "origen": self.origen,
            "destino": self.destino,
            "distancia_km": self.distancia_km,
            "tiempo_h": self.tiempo_h,
            "costo_cop": self.costo_cop,
            "riesgo_termico": self.riesgo_termico,
            "estado_via": self.estado_via,
            "tipo_via": self.tipo_via,
            "capacidad_kg": self.capacidad_kg,
            "num_transbordos": self.num_transbordos,
            "peso_combinado": self.peso_default,
            "perdida_esperada": self.perdida_esperada,
        }

    def __repr__(self) -> str:
        return (
            f"Arista({self.origen} → {self.destino}, "
            f"{self.distancia_km:.1f}km, {self.tiempo_h:.2f}h, "
            f"peso={self.peso_default:.4f})"
        )

