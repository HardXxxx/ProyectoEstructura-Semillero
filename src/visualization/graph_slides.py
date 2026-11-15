"""
graph_slides.py — Presentación interactiva por diapositivas para explorar:
  Diapositiva 1: Mapa Geográfico Interactivo (Folium + OSRM + Ríos)
  Diapositiva 2: Grafo Vial (Subgrafo Terrestre y Acceso Local)
  Diapositiva 3: Grafo de Ríos (Subgrafo Fluvial y Red Hidrográfica Topológica)
  Diapositiva 4: Grafo Mixto Multimodal Completo (25 vértices, 41 aristas, itinerarios y lista de adyacencia)

Permite navegar hacia adelante ("Siguiente →") y hacia atrás ("← Anterior"),
arrastrar nodos, hacer zoom/paneo, filtrar por categoría y hacer clic sobre
cualquier vértice o arista para inspeccionar todas las propiedades y variables
que almacena en memoria.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Any

from ..models.grafo_logistico import GrafoLogistico
from ..utils.constants import (
    ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA,
    TIEMPO_TRANSBORDO_MUELLE_H, FACTOR_DETERIORO_ZONA_D,
)


def construir_datos_slides(
    grafo: GrafoLogistico,
    geo_cache: Optional[Dict[str, list]] = None,
    itinerarios: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Serializa todos los vértices, aristas, itinerarios, lista de adyacencia
    y estadísticas del grafo fluvial para el visor interactivo por diapositivas.
    """
    geo_cache = geo_cache or {}
    itinerarios = list(itinerarios or [])

    # 1. Grados de entrada y salida por nodo
    grado_entrada: Dict[str, int] = {n.id: 0 for n in grafo.nodos}
    grado_salida: Dict[str, int] = {n.id: 0 for n in grafo.nodos}
    vecinos_salida: Dict[str, List[str]] = {n.id: [] for n in grafo.nodos}
    vecinos_entrada: Dict[str, List[str]] = {n.id: [] for n in grafo.nodos}

    for ar in grafo.aristas:
        grado_salida[ar.origen] = grado_salida.get(ar.origen, 0) + 1
        grado_entrada[ar.destino] = grado_entrada.get(ar.destino, 0) + 1
        vecinos_salida.setdefault(ar.origen, []).append(ar.destino)
        vecinos_entrada.setdefault(ar.destino, []).append(ar.origen)

    # 2. Serializar Nodos (Vértices) con todos sus atributos y metadatos
    nodos_data = []
    for n in grafo.nodos:
        d = n.to_dict()
        d["es_candidato_frio"] = n.es_candidato_frio
        d["es_origen"] = n.es_origen
        d["es_destino"] = n.es_destino
        d["grado_entrada"] = grado_entrada.get(n.id, 0)
        d["grado_salida"] = grado_salida.get(n.id, 0)
        d["vecinos_salida"] = vecinos_salida.get(n.id, [])
        d["vecinos_entrada"] = vecinos_entrada.get(n.id, [])
        d["metadata"] = n.metadata or {}
        if n.tipo == "M":
            d["efecto_vertice"] = (
                f"Transbordo fluvial → terrestre: añade +{TIEMPO_TRANSBORDO_MUELLE_H:.2f} h "
                "al tiempo sin frío y suma +1 transbordo."
            )
        elif n.tipo == "D":
            d["efecto_vertice"] = (
                f"Zona crítica de deterioro térmico: multiplica ×{FACTOR_DETERIORO_ZONA_D} "
                "la tasa de pérdida del tramo entrante."
            )
        elif n.tipo == "H":
            d["efecto_vertice"] = (
                "Hábitat protegido (restriccion_ambiental=True): bloqueado como nodo "
                "intermedio en Dijkstra, Bellman-Ford y Floyd-Warshall."
            )
        elif n.tipo == "F":
            d["efecto_vertice"] = (
                f"Planta de refrigeración ({n.capacidad_frio_kg:,.0f} kg): abate la "
                "temperatura a 0 °C y reinicia tiempo_sin_frio = 0.0 h."
            )
        elif n.tipo == "A":
            d["efecto_vertice"] = (
                f"Centro de acopio obligatorio (ADR-06, {n.capacidad_frio_kg:,.0f} kg): "
                + ("con energía y pre-frío (reinicia tiempo_sin_frio)."
                   if n.tiene_energia else "sin energía eléctrica confiable (no candidato FLP).")
            )
        elif n.tipo == "P":
            d["efecto_vertice"] = (
                f"Nodo fuente de captura artesanal: inyecta {n.produccion_kg_dia:.2f} kg/día "
                "a temperatura ambiente (30 °C)."
            )
        else:
            d["efecto_vertice"] = (
                f"Mercado / Punto de venta final (demanda: {n.metadata.get('demanda_kg_dia', 0)} kg/día)."
            )
        nodos_data.append(d)

    # 3. Serializar Aristas con las 22 variables y el desglose de la fórmula multicriterio
    aristas_data = []
    for idx, ar in enumerate(grafo.aristas, start=1):
        ad = ar.to_dict()
        ad["idx"] = idx
        ad["id"] = f"{ar.origen}->{ar.destino}"
        ad["factor_estado_via"] = ar.factor_estado_via
        ad["factor_flujo"] = ar.factor_flujo
        clave_geo = f"{ar.origen}-{ar.destino}"
        pts_geo = geo_cache.get(clave_geo) or []
        ad["num_puntos_geometria"] = len(pts_geo)
        # Muestra compactada de puntos para dibujar trazado real en el canvas fluvial/mixto
        if len(pts_geo) > 40:
            paso = max(1, len(pts_geo) // 35)
            muestra = pts_geo[::paso]
            if muestra[-1] != pts_geo[-1]:
                muestra.append(pts_geo[-1])
            ad["geometria_muestra"] = muestra
        else:
            ad["geometria_muestra"] = pts_geo

        # Desglose exacto de los 6 términos de w_ij
        t_term = ALPHA * (ar.tiempo_efectivo_h / 48.0)
        c_term = BETA * (ar.costo_total_cop / 5_000_000.0)
        r_term = GAMMA * ar.riesgo_combinado
        p_term = DELTA * (ar.perdida_esperada / 100.0)
        conf_term = EPSILON * (1.0 - ar.confiabilidad_via)
        ret_term = ZETA * ar.prob_retraso
        ad["desglose_peso"] = {
            "t_term": round(t_term, 6),
            "c_term": round(c_term, 6),
            "r_term": round(r_term, 6),
            "p_term": round(p_term, 6),
            "conf_term": round(conf_term, 6),
            "ret_term": round(ret_term, 6),
        }
        aristas_data.append(ad)

    # 4. Serializar Itinerarios
    itinerarios_data = []
    for it in itinerarios:
        itinerarios_data.append({
            "origen_id": it.origen_id,
            "acopio_id": it.acopio_id,
            "centro_frio_id": it.centro_frio_id,
            "destino_id": it.destino_id,
            "distancia_total_km": round(it.distancia_total_km, 2),
            "tiempo_total_h": round(it.tiempo_total_h, 2),
            "perdida_pct": round(it.perdida_pct, 2),
            "kg_inicial": round(it.kg_inicial, 2),
            "kg_final": round(it.kg_final, 2),
            "nodos_restringidos_evitados": list(it.nodos_restringidos_evitados or []),
            "ruta": list(it.ruta or []),
            "tramos": [
                {
                    "origen": t.origen,
                    "destino": t.destino,
                    "tipo_via": t.tipo_via,
                    "distancia_km": t.distancia_km,
                    "tiempo_h": t.tiempo_h,
                    "tipo_nodo_destino": t.tipo_nodo_destino,
                    "efecto": t.efecto,
                }
                for t in getattr(it, "tramos", [])
            ],
        })

    # 5. Estadísticas y trazos simplificados de la Red Fluvial Topológica (rios.shp)
    rios_lineas = []
    stats_rio = {"vertices": 12413, "aristas_dirigidas": 24790, "celdas_rejilla": 447}
    try:
        from .river_network import obtener_capa_rios_geojson
        capa_rios = obtener_capa_rios_geojson(tolerancia=0.002)
        if capa_rios and capa_rios.get("features"):
            for feat in capa_rios["features"]:
                coords = feat.get("geometry", {}).get("coordinates", [])
                if len(coords) >= 2:
                    rios_lineas.append(coords)
    except Exception:
        pass

    return {
        "nodos": nodos_data,
        "aristas": aristas_data,
        "itinerarios": itinerarios_data,
        "rios_lineas": rios_lineas,
        "stats_rio": stats_rio,
    }


def generar_html_slides(
    grafo: GrafoLogistico,
    geo_cache: Optional[Dict[str, list]] = None,
    itinerarios: Optional[list] = None,
) -> str:
    """
    Devuelve el bloque HTML + CSS + JS autocontenido que añade la navegación
    por diapositivas (Anterior / Siguiente) y los visores interactivos de:
      - Diapositiva 1: Mapa Geográfico (Folium)
      - Diapositiva 2: Grafo Vial (Terrestre + Acceso Local)
      - Diapositiva 3: Grafo de Ríos (Fluvial + Red Topológica rios.shp)
      - Diapositiva 4: Grafo Mixto Multimodal Completo
    """
    payload = construir_datos_slides(grafo, geo_cache, itinerarios)
    json_payload = json.dumps(payload, ensure_ascii=False)

    return f"""
<!-- ═══════════════════════════════════════════════════════════════════════
     SISTEMA INTERACTIVO DE DIAPOSITIVAS Y EXPLORADOR DE GRAFOS
     ═══════════════════════════════════════════════════════════════════════ -->
<style>
  /* Barra superior de navegación de diapositivas */
  #agy-slide-nav {{
    position: fixed;
    top: 12px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 10000;
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(15, 23, 42, 0.94);
    backdrop-filter: blur(10px);
    color: #f8fafc;
    padding: 8px 14px;
    border-radius: 999px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(255, 255, 255, 0.12);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    user-select: none;
  }}

  .agy-nav-btn {{
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #fff;
    border: none;
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    transition: all 0.18s ease;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4);
  }}
  .agy-nav-btn:hover:not(:disabled) {{
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    transform: translateY(-1px);
  }}
  .agy-nav-btn:disabled {{
    background: #334155;
    color: #64748b;
    cursor: not-allowed;
    box-shadow: none;
  }}

  .agy-slide-tabs {{
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 0 6px;
  }}

  .agy-slide-tab {{
    background: transparent;
    color: #cbd5e1;
    border: 1px solid transparent;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.18s ease;
    white-space: nowrap;
  }}
  .agy-slide-tab:hover {{
    color: #fff;
    background: rgba(255, 255, 255, 0.08);
  }}
  .agy-slide-tab.active {{
    background: rgba(56, 189, 248, 0.18);
    color: #38bdf8;
    border-color: rgba(56, 189, 248, 0.45);
    box-shadow: inset 0 0 12px rgba(56, 189, 248, 0.15);
  }}

  /* Contenedor de las diapositivas de grafos (2, 3 y 4) */
  #agy-graph-deck {{
    position: fixed;
    inset: 0;
    z-index: 9500;
    background: radial-gradient(circle at 20% 20%, #0f172a 0%, #090d16 100%);
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    display: none;
    flex-direction: column;
    padding-top: 68px;
    box-sizing: border-box;
  }}
  #agy-graph-deck.visible {{
    display: flex;
  }}

  /* Encabezado descriptivo de la diapositiva activa */
  .agy-deck-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 24px;
    background: rgba(30, 41, 59, 0.65);
    border-bottom: 1px solid rgba(148, 163, 184, 0.18);
    flex-wrap: wrap;
    gap: 12px;
  }}
  .agy-deck-title-box h2 {{
    margin: 0;
    font-size: 18px;
    font-weight: 700;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .agy-deck-title-box p {{
    margin: 3px 0 0 0;
    font-size: 12.5px;
    color: #94a3b8;
  }}

  .agy-deck-controls {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }}
  .agy-ctrl-btn, .agy-ctrl-select {{
    background: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .agy-ctrl-btn:hover, .agy-ctrl-select:hover {{
    border-color: #38bdf8;
    color: #fff;
  }}
  .agy-ctrl-btn.active {{
    background: #0284c7;
    border-color: #38bdf8;
    color: #fff;
  }}

  /* Cuerpo principal: Lienzo interactivo a la izquierda + Inspector a la derecha */
  .agy-deck-body {{
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 410px;
    min-height: 0;
    overflow: hidden;
  }}

  .agy-canvas-wrap {{
    position: relative;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background:
      linear-gradient(rgba(148, 163, 184, 0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(148, 163, 184, 0.04) 1px, transparent 1px);
    background-size: 32px 32px;
  }}

  #agy-graph-svg {{
    width: 100%;
    height: 100%;
    cursor: grab;
    display: block;
  }}
  #agy-graph-svg:active {{
    cursor: grabbing;
  }}

  /* Tarjetas KPI flotantes sobre el lienzo */
  .agy-kpi-strip {{
    position: absolute;
    top: 14px;
    left: 16px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    pointer-events: none;
  }}
  .agy-kpi-card {{
    background: rgba(15, 23, 42, 0.86);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 10px;
    padding: 8px 13px;
    backdrop-filter: blur(6px);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
  }}
  .agy-kpi-val {{
    font-size: 16px;
    font-weight: 700;
    color: #38bdf8;
  }}
  .agy-kpi-lbl {{
    font-size: 10.5px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }}

  /* Leyenda interactiva inferior del lienzo */
  .agy-canvas-legend {{
    position: absolute;
    bottom: 14px;
    left: 16px;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 11.5px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    max-width: calc(100% - 40px);
  }}
  .agy-leg-item {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #cbd5e1;
  }}
  .agy-leg-dot {{
    width: 11px;
    height: 11px;
    border-radius: 50%;
    display: inline-block;
  }}
  .agy-leg-line {{
    width: 22px;
    height: 0;
    display: inline-block;
  }}

  /* Ayuda rápida en esquina inferior derecha del lienzo */
  .agy-canvas-hint {{
    position: absolute;
    bottom: 14px;
    right: 16px;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #bae6fd;
    padding: 7px 12px;
    border-radius: 8px;
    font-size: 11.5px;
    pointer-events: none;
  }}

  /* Panel lateral derecho: Inspector de Vértices y Aristas */
  .agy-inspector {{
    background: #0f172a;
    border-left: 1px solid rgba(148, 163, 184, 0.2);
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }}
  .agy-inspector-header {{
    padding: 14px 18px;
    background: rgba(30, 41, 59, 0.85);
    border-bottom: 1px solid rgba(148, 163, 184, 0.18);
  }}
  .agy-inspector-header h3 {{
    margin: 0;
    font-size: 14.5px;
    color: #f8fafc;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .agy-inspector-content {{
    flex: 1;
    overflow-y: auto;
    padding: 16px 18px;
    font-size: 12.5px;
    line-height: 1.5;
  }}

  /* Tarjetas dentro del inspector */
  .agy-prop-group {{
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 12px;
  }}
  .agy-prop-group h4 {{
    margin: 0 0 8px 0;
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #38bdf8;
    border-bottom: 1px solid rgba(148, 163, 184, 0.14);
    padding-bottom: 5px;
  }}
  .agy-prop-row {{
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px dashed rgba(148, 163, 184, 0.1);
    gap: 8px;
  }}
  .agy-prop-row:last-child {{
    border-bottom: none;
  }}
  .agy-prop-key {{
    color: #94a3b8;
    font-family: Consolas, monospace;
    font-size: 11.5px;
  }}
  .agy-prop-val {{
    color: #f1f5f9;
    font-weight: 600;
    text-align: right;
  }}

  .agy-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
  }}

  .agy-effect-box {{
    background: rgba(56, 189, 248, 0.1);
    border-left: 3px solid #38bdf8;
    padding: 9px 11px;
    border-radius: 6px;
    color: #e0f2fe;
    font-size: 12px;
    margin-bottom: 12px;
  }}

  .agy-code-block {{
    background: #090d16;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 10px;
    font-family: Consolas, Monaco, monospace;
    font-size: 11px;
    color: #a5f3fc;
    overflow-x: auto;
    white-space: pre;
  }}

  .agy-chip-list {{
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 4px;
  }}
  .agy-chip {{
    background: #1e293b;
    border: 1px solid #475569;
    color: #e2e8f0;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
  }}
  .agy-chip:hover {{
    border-color: #38bdf8;
    background: #0f172a;
    color: #38bdf8;
  }}

  /* Barras de progreso de la fórmula multicriterio */
  .agy-bar-wrap {{
    margin: 5px 0;
  }}
  .agy-bar-label {{
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #cbd5e1;
  }}
  .agy-bar-bg {{
    height: 6px;
    background: #1e293b;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 2px;
  }}
  .agy-bar-fill {{
    height: 100%;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    border-radius: 4px;
  }}

  /* Estilos de elementos SVG del grafo */
  .agy-edge-path {{
    fill: none;
    cursor: pointer;
    transition: stroke-width 0.15s, opacity 0.15s;
  }}
  .agy-edge-path:hover {{
    stroke-width: 5.5px !important;
    opacity: 1 !important;
  }}
  .agy-edge-hit {{
    fill: none;
    stroke: transparent;
    stroke-width: 16px;
    cursor: pointer;
  }}
  .agy-node-group {{
    cursor: pointer;
  }}
  .agy-node-circle {{
    transition: r 0.15s, stroke-width 0.15s;
  }}
  .agy-node-group:hover .agy-node-circle {{
    stroke-width: 4px;
    filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.85));
  }}

  @keyframes agyDashFlow {{
    to {{
      stroke-dashoffset: -40;
    }}
  }}
  .agy-route-highlight {{
    animation: agyDashFlow 1.2s linear infinite;
  }}
</style>

<!-- Barra de navegación persistente tipo diapositivas -->
<div id="agy-slide-nav">
  <button id="agy-btn-prev" class="agy-nav-btn" onclick="agyCambiarSlide(agySlideActual - 1)" disabled>
    ◀ Anterior
  </button>
  <div class="agy-slide-tabs">
    <button class="agy-slide-tab active" data-slide="0" onclick="agyCambiarSlide(0)">
      1. 🗺️ Mapa Geográfico
    </button>
    <button class="agy-slide-tab" data-slide="1" onclick="agyCambiarSlide(1)">
      2. 🛣️ Grafo Vial
    </button>
    <button class="agy-slide-tab" data-slide="2" onclick="agyCambiarSlide(2)">
      3. 🌊 Grafo de Ríos
    </button>
    <button class="agy-slide-tab" data-slide="3" onclick="agyCambiarSlide(3)">
      4. 🔗 Grafo Mixto Multimodal
    </button>
  </div>
  <button id="agy-btn-next" class="agy-nav-btn" onclick="agyCambiarSlide(agySlideActual + 1)">
    Siguiente ▶
  </button>
</div>

<!-- Contenedor interactivo para las Diapositivas 2, 3 y 4 -->
<div id="agy-graph-deck">
  <div class="agy-deck-header">
    <div class="agy-deck-title-box">
      <h2 id="agy-slide-title">🛣️ Diapositiva 2: Grafo Vial</h2>
      <p id="agy-slide-subtitle">Haz clic en cualquier vértice o arista para inspeccionar su estructura de datos y sus 22 variables.</p>
    </div>
    <div class="agy-deck-controls">
      <button id="agy-layout-geo" class="agy-ctrl-btn" onclick="agyCambiarLayout('geo')">
        📍 Coordenadas Geográficas
      </button>
      <button id="agy-layout-hier" class="agy-ctrl-btn active" onclick="agyCambiarLayout('hier')">
        🧬 Vista Jerárquica (P → M → A → F → V)
      </button>
      <select id="agy-itin-select" class="agy-ctrl-select" onchange="agySeleccionarItinerario(this.value)">
        <option value="">✨ Resaltar Itinerario de Pesca...</option>
      </select>
      <select id="agy-quick-select" class="agy-ctrl-select" onchange="agySeleccionarElementoRapido(this.value)">
        <option value="">🔍 Ir a Vértice / Arista...</option>
      </select>
      <button class="agy-ctrl-btn" onclick="agyResetVista()">
        🎯 Centrar Vista
      </button>
    </div>
  </div>

  <div class="agy-deck-body">
    <!-- Lienzo SVG Interactivo -->
    <div class="agy-canvas-wrap" id="agy-canvas-wrap">
      <div class="agy-kpi-strip" id="agy-kpi-strip"></div>

      <svg id="agy-graph-svg" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker id="arrow-terrestre" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#f97316"/>
          </marker>
          <marker id="arrow-acceso" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#94a3b8"/>
          </marker>
          <marker id="arrow-fluvial" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8"/>
          </marker>
          <marker id="arrow-active" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#facc15"/>
          </marker>
        </defs>
        <g id="agy-viewport">
          <g id="agy-layer-rivers"></g>
          <g id="agy-layer-edges"></g>
          <g id="agy-layer-nodes"></g>
        </g>
      </svg>

      <div class="agy-canvas-legend">
        <span class="agy-leg-item"><span class="agy-leg-dot" style="background:#2196F3"></span>Pesca (P)</span>
        <span class="agy-leg-item"><span class="agy-leg-dot" style="background:#9C27B0"></span>Muelle (M)</span>
        <span class="agy-leg-item"><span class="agy-leg-dot" style="background:#FF9800"></span>Acopio (A)</span>
        <span class="agy-leg-item"><span class="agy-leg-dot" style="background:#00BCD4"></span>Frío (F)</span>
        <span class="agy-leg-item"><span class="agy-leg-dot" style="background:#F44336"></span>Deterioro (D)</span>
        <span class="agy-leg-item"><span class="agy-leg-dot" style="background:#4CAF50"></span>Hábitat (H)</span>
        <span class="agy-leg-item"><span class="agy-leg-dot" style="background:#E91E63"></span>Venta (V)</span>
        <span class="agy-leg-item"><span class="agy-leg-line" style="border-top:3px dashed #38bdf8"></span>Vía Fluvial</span>
        <span class="agy-leg-item"><span class="agy-leg-line" style="border-top:3px solid #94a3b8"></span>Acceso Local</span>
        <span class="agy-leg-item"><span class="agy-leg-line" style="border-top:3px solid #f97316"></span>Carretera Terrestre</span>
      </div>

      <div class="agy-canvas-hint">
        💡 <b>Interactivo:</b> Clic en nodo/arista = ver propiedades · Arrastrar nodo = mover · Rueda = zoom · Teclas ← / → = cambiar diapositiva
      </div>
    </div>

    <!-- Inspector de Propiedades de Vértices y Aristas -->
    <div class="agy-inspector">
      <div class="agy-inspector-header">
        <h3>
          <span id="agy-insp-title">🔎 Inspector de Estructura de Datos</span>
          <span id="agy-insp-badge" class="agy-badge" style="background:#1e293b;color:#38bdf8;">Grafo Activo</span>
        </h3>
      </div>
      <div class="agy-inspector-content" id="agy-insp-body"></div>
    </div>
  </div>
</div>

<script>
const AGY_DATA = {json_payload};

const AGY_COLORES_NODO = {{
  P: "#2196F3",
  M: "#9C27B0",
  A: "#FF9800",
  F: "#00BCD4",
  D: "#F44336",
  H: "#4CAF50",
  V: "#E91E63"
}};

let agySlideActual = 0;
let agyLayoutModo = "hier"; // 'hier' | 'geo'
let agySeleccion = null;    // {{ tipo: 'nodo'|'arista', id: ... }}
let agyItinerarioActivo = "";
let agyPosiciones = {{}};
let agyTransform = {{ x: 0, y: 0, k: 1 }};
let agyDraggingNode = null;
let agyPanning = false;
let agyPanStart = {{ x: 0, y: 0 }};

// Poblar selectores de itinerarios
(function initSelectores() {{
  const selIt = document.getElementById("agy-itin-select");
  (AGY_DATA.itinerarios || []).forEach(it => {{
    const opt = document.createElement("option");
    opt.value = it.origen_id;
    opt.textContent = `🐟 Itinerario ${{it.origen_id}}: ${{it.ruta.join(" → ")}} (${{it.distancia_total_km}} km)`;
    selIt.appendChild(opt);
  }});
}})();

// Filtrar nodos y aristas según la diapositiva activa:
//   Slide 1: Grafo Vial (terrestre + acceso_local)
//   Slide 2: Grafo de Ríos (fluvial)
//   Slide 3: Grafo Mixto Multimodal Completo (todas)
function agySubgrafoActual() {{
  let aristas = [];
  if (agySlideActual === 1) {{
    aristas = AGY_DATA.aristas.filter(a => a.tipo_via === "terrestre" || a.tipo_via === "acceso_local");
  }} else if (agySlideActual === 2) {{
    aristas = AGY_DATA.aristas.filter(a => a.tipo_via === "fluvial");
  }} else {{
    aristas = AGY_DATA.aristas.slice();
  }}
  const idsActivos = new Set();
  aristas.forEach(a => {{
    idsActivos.add(a.origen);
    idsActivos.add(a.destino);
  }});
  const nodos = AGY_DATA.nodos.filter(n => agySlideActual === 3 || idsActivos.has(n.id));
  return {{ nodos, aristas }};
}}

// Calcular coordenadas (x, y) en el lienzo según modo Jerárquico o Geográfico
function agyCalcularPosiciones() {{
  const svg = document.getElementById("agy-graph-svg");
  const W = svg.clientWidth || 960;
  const H = svg.clientHeight || 640;
  const padX = 85, padY = 70;

  if (agyLayoutModo === "geo") {{
    // Escalar lat/lon con compresión suave de longitudes lejanas (Villavicencio/Bogotá)
    // para que tanto la zona local de Puerto Gaitán como Bogotá se vean claros
    const lats = AGY_DATA.nodos.map(n => n.latitud);
    const lons = AGY_DATA.nodos.map(n => n.longitud);
    const minLat = Math.min(...lats), maxLat = Math.max(...lats);
    const minLon = Math.min(...lons), maxLon = Math.max(...lons);

    AGY_DATA.nodos.forEach(n => {{
      const nx = (n.longitud - minLon) / ((maxLon - minLon) || 1);
      const ny = 1 - (n.latitud - minLat) / ((maxLat - minLat) || 1);
      agyPosiciones[n.id] = {{
        x: padX + nx * (W - 2 * padX),
        y: padY + ny * (H - 2 * padY)
      }};
    }});
  }} else {{
    // Disposición jerárquica por etapas de la cadena logística:
    // Col 0: P (Pesca) -> Col 1: M (Muelles) -> Col 2: A (Acopios)
    // Col 3: D / H (Zonas críticas y hábitats) -> Col 4: F (Frío) -> Col 5: V (Mercados)
    const columnas = {{
      P: 0,
      M: 1,
      A: 2,
      D: 3,
      H: 3,
      F: 4,
      V: 5
    }};
    const grupos = {{ 0: [], 1: [], 2: [], 3: [], 4: [], 5: [] }};
    AGY_DATA.nodos.forEach(n => {{
      const c = columnas[n.tipo] !== undefined ? columnas[n.tipo] : 3;
      grupos[c].push(n);
    }});
    Object.keys(grupos).forEach(c => {{
      const lista = grupos[c];
      const x = padX + (Number(c) / 5) * (W - 2 * padX);
      lista.forEach((n, idx) => {{
        const y = padY + ((idx + 0.5) / Math.max(lista.length, 1)) * (H - 2 * padY);
        agyPosiciones[n.id] = {{ x, y }};
      }});
    }});
  }}
}}

function agyActualizarTransform() {{
  const vp = document.getElementById("agy-viewport");
  if (vp) {{
    vp.setAttribute("transform", `translate(${{agyTransform.x}}, ${{agyTransform.y}}) scale(${{agyTransform.k}})`);
  }}
}}

function agyResetVista() {{
  agyTransform = {{ x: 0, y: 0, k: 1 }};
  agyCalcularPosiciones();
  agyRenderizarGrafo();
}}

function agyCambiarLayout(modo) {{
  agyLayoutModo = modo;
  document.getElementById("agy-layout-geo").classList.toggle("active", modo === "geo");
  document.getElementById("agy-layout-hier").classList.toggle("active", modo === "hier");
  agyResetVista();
}}

function agyCambiarSlide(nuevoSlide) {{
  if (nuevoSlide < 0 || nuevoSlide > 3) return;
  agySlideActual = nuevoSlide;

  document.getElementById("agy-btn-prev").disabled = (agySlideActual === 0);
  document.getElementById("agy-btn-next").disabled = (agySlideActual === 3);

  document.querySelectorAll(".agy-slide-tab").forEach(btn => {{
    btn.classList.toggle("active", Number(btn.dataset.slide) === agySlideActual);
  }});

  const deck = document.getElementById("agy-graph-deck");
  if (agySlideActual === 0) {{
    deck.classList.remove("visible");
    return;
  }}

  deck.classList.add("visible");

  // Configurar títulos según la diapositiva
  const titulos = {{
    1: [
      "🛣️ Diapositiva 2: Grafo Vial (Subgrafo Terrestre y de Acceso Local)",
      "Representa las 31 conexiones viales (19 carreteras nacionales/secundarias y 12 vías terciarias de acceso local) y sus 22 variables."
    ],
    2: [
      "🌊 Diapositiva 3: Grafo de Ríos (Subgrafo Fluvial y Red Hidrográfica Topológica)",
      "Representa las 10 aristas fluviales navegables entre puntos de pesca, muelles y acopios sobre el grafo de 12,413 vértices de rios.shp."
    ],
    3: [
      "🔗 Diapositiva 4: Grafo Mixto Multimodal Completo (Vial + Fluvial + M/D/H/A/F/V)",
      "Integra los 25 vértices y las 41 aristas dirigidas de la cadena de frío. Haz clic en cualquier nodo o arista para inspeccionar su objeto en memoria."
    ]
  }};
  document.getElementById("agy-slide-title").textContent = titulos[agySlideActual][0];
  document.getElementById("agy-slide-subtitle").textContent = titulos[agySlideActual][1];

  // En el grafo de ríos, si el usuario quiere ver la red geográfica de fondo, funciona en ambos layouts
  agyCalcularPosiciones();
  agyActualizarSelectorRapido();
  agyRenderizarGrafo();
  agyMostrarResumenSubgrafo();
}}

function agyActualizarSelectorRapido() {{
  const {{ nodos, aristas }} = agySubgrafoActual();
  const sel = document.getElementById("agy-quick-select");
  sel.innerHTML = '<option value="">🔍 Ir a Vértice / Arista...</option>';

  const grpN = document.createElement("optgroup");
  grpN.label = `Vértices en esta vista (${{nodos.length}})`;
  nodos.forEach(n => {{
    const o = document.createElement("option");
    o.value = `N:${{n.id}}`;
    o.textContent = `[${{n.id}}] ${{n.nombre}} (${{n.tipo_descripcion}})`;
    grpN.appendChild(o);
  }});
  sel.appendChild(grpN);

  const grpA = document.createElement("optgroup");
  grpA.label = `Aristas en esta vista (${{aristas.length}})`;
  aristas.forEach(a => {{
    const o = document.createElement("option");
    o.value = `E:${{a.id}}`;
    o.textContent = `#${{String(a.idx).padStart(2,'0')}} ${{a.origen}} → ${{a.destino}} (${{a.tipo_via}}, ${{a.distancia_km}} km, w=${{a.peso_combinado.toFixed(4)}})`;
    grpA.appendChild(o);
  }});
  sel.appendChild(grpA);
}}

function agySeleccionarElementoRapido(val) {{
  if (!val) return;
  const [tipo, id] = val.split(":");
  if (tipo === "N") {{
    const n = AGY_DATA.nodos.find(x => x.id === id);
    if (n) agyInspeccionarNodo(n);
  }} else if (tipo === "E") {{
    const a = AGY_DATA.aristas.find(x => x.id === id);
    if (a) agyInspeccionarArista(a);
  }}
}}

function agySeleccionarItinerario(origenId) {{
  agyItinerarioActivo = origenId;
  agyRenderizarGrafo();
  if (origenId) {{
    const it = (AGY_DATA.itinerarios || []).find(x => x.origen_id === origenId);
    if (it) agyInspeccionarItinerario(it);
  }} else {{
    agyMostrarResumenSubgrafo();
  }}
}}

// Renderiza el SVG interactivo con ríos de fondo, aristas curvas dirigidas y nodos arrastrables
function agyRenderizarGrafo() {{
  const {{ nodos, aristas }} = agySubgrafoActual();
  const layerRivers = document.getElementById("agy-layer-rivers");
  const layerEdges = document.getElementById("agy-layer-edges");
  const layerNodes = document.getElementById("agy-layer-nodes");
  layerRivers.innerHTML = "";
  layerEdges.innerHTML = "";
  layerNodes.innerHTML = "";

  // Actualizar KPIs superiores
  const kpiStrip = document.getElementById("agy-kpi-strip");
  const kmTotal = aristas.reduce((s, a) => s + a.distancia_km, 0);
  const fluvCount = aristas.filter(a => a.tipo_via === "fluvial").length;
  const vialCount = aristas.filter(a => a.tipo_via !== "fluvial").length;
  kpiStrip.innerHTML = `
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{nodos.length}}</div>
      <div class="agy-kpi-lbl">Vértices Activos</div>
    </div>
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{aristas.length}}</div>
      <div class="agy-kpi-lbl">Aristas Dirigidas</div>
    </div>
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{vialCount}} / ${{fluvCount}}</div>
      <div class="agy-kpi-lbl">Viales / Fluviales</div>
    </div>
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{kmTotal.toFixed(1)}} km</div>
      <div class="agy-kpi-lbl">Longitud Acumulada</div>
    </div>
    ${{agySlideActual === 2 ? `
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{AGY_DATA.stats_rio.vertices.toLocaleString()}}</div>
      <div class="agy-kpi-lbl">Vértices rios.shp</div>
    </div>` : ""}}
  `;

  // Si estamos en modo geográfico y en el Grafo de Ríos o Mixto, dibujar la malla hidrográfica de fondo
  if (agyLayoutModo === "geo" && (agySlideActual === 2 || agySlideActual === 3)) {{
    const svg = document.getElementById("agy-graph-svg");
    const W = svg.clientWidth || 960;
    const H = svg.clientHeight || 640;
    const padX = 85, padY = 70;
    const lats = AGY_DATA.nodos.map(n => n.latitud);
    const lons = AGY_DATA.nodos.map(n => n.longitud);
    const minLat = Math.min(...lats), maxLat = Math.max(...lats);
    const minLon = Math.min(...lons), maxLon = Math.max(...lons);

    (AGY_DATA.rios_lineas || []).forEach(coords => {{
      const pts = coords.map(c => {{
        const x = padX + ((c[0] - minLon) / ((maxLon - minLon) || 1)) * (W - 2 * padX);
        const y = padY + (1 - (c[1] - minLat) / ((maxLat - minLat) || 1)) * (H - 2 * padY);
        return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
      }}).join(" ");
      const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      poly.setAttribute("points", pts);
      poly.setAttribute("fill", "none");
      poly.setAttribute("stroke", "#0284c7");
      poly.setAttribute("stroke-width", agySlideActual === 2 ? "1.6" : "1.0");
      poly.setAttribute("stroke-opacity", agySlideActual === 2 ? "0.38" : "0.22");
      layerRivers.appendChild(poly);
    }});
  }}

  // Conjunto de aristas resaltadas si hay un itinerario activo
  const aristasItinerario = new Set();
  const nodosItinerario = new Set();
  if (agyItinerarioActivo) {{
    const it = (AGY_DATA.itinerarios || []).find(x => x.origen_id === agyItinerarioActivo);
    if (it) {{
      (it.ruta || []).forEach(nid => nodosItinerario.add(nid));
      for (let i = 0; i < it.ruta.length - 1; i++) {{
        aristasItinerario.add(`${{it.ruta[i]}}->${{it.ruta[i + 1]}}`);
      }}
    }}
  }}

  // Dibujar Aristas
  aristas.forEach(a => {{
    const p1 = agyPosiciones[a.origen];
    const p2 = agyPosiciones[a.destino];
    if (!p1 || !p2) return;

    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const dist = Math.hypot(dx, dy) || 1;
    // Curvatura suave para distinguir aristas y mejorar legibilidad
    const offset = Math.min(26, dist * 0.12);
    const mx = (p1.x + p2.x) / 2 - (dy / dist) * offset;
    const my = (p1.y + p2.y) / 2 + (dx / dist) * offset;
    const pathD = `M ${{p1.x}} ${{p1.y}} Q ${{mx}} ${{my}} ${{p2.x}} ${{p2.y}}`;

    const esSeleccionada = agySeleccion && agySeleccion.tipo === "arista" && agySeleccion.id === a.id;
    const esDeItinerario = aristasItinerario.has(a.id);

    let stroke = "#f97316";
    let marker = "url(#arrow-terrestre)";
    let dash = "";
    if (a.tipo_via === "fluvial") {{
      stroke = "#38bdf8";
      marker = "url(#arrow-fluvial)";
      dash = "7,5";
    }} else if (a.tipo_via === "acceso_local") {{
      stroke = "#94a3b8";
      marker = "url(#arrow-acceso)";
      dash = a.estado_via === "regular" ? "4,4" : "";
    }} else {{
      stroke = a.estado_via === "mala" ? "#ef4444" : (a.estado_via === "regular" ? "#f97316" : "#22c55e");
    }}

    if (esSeleccionada || esDeItinerario) {{
      stroke = "#facc15";
      marker = "url(#arrow-active)";
    }}

    const gEdge = document.createElementNS("http://www.w3.org/2000/svg", "g");

    const pathVis = document.createElementNS("http://www.w3.org/2000/svg", "path");
    pathVis.setAttribute("d", pathD);
    pathVis.setAttribute("class", "agy-edge-path" + (esDeItinerario ? " agy-route-highlight" : ""));
    pathVis.setAttribute("stroke", stroke);
    pathVis.setAttribute("stroke-width", (esSeleccionada || esDeItinerario) ? "4.5" : "2.4");
    pathVis.setAttribute("stroke-opacity", (agyItinerarioActivo && !esDeItinerario && !esSeleccionada) ? "0.25" : "0.88");
    if (dash || esDeItinerario) {{
      pathVis.setAttribute("stroke-dasharray", esDeItinerario ? "10,6" : dash);
    }}
    pathVis.setAttribute("marker-end", marker);

    const pathHit = document.createElementNS("http://www.w3.org/2000/svg", "path");
    pathHit.setAttribute("d", pathD);
    pathHit.setAttribute("class", "agy-edge-hit");
    pathHit.addEventListener("click", (ev) => {{
      ev.stopPropagation();
      agyInspeccionarArista(a);
    }});

    // Etiqueta de la arista en el punto medio de la curva cuadrática
    const lx = 0.25 * p1.x + 0.5 * mx + 0.25 * p2.x;
    const ly = 0.25 * p1.y + 0.5 * my + 0.25 * p2.y;
    const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    txt.setAttribute("x", lx);
    txt.setAttribute("y", ly - 4);
    txt.setAttribute("text-anchor", "middle");
    txt.setAttribute("fill", (esSeleccionada || esDeItinerario) ? "#fef08a" : "#cbd5e1");
    txt.setAttribute("font-size", "9.5px");
    txt.setAttribute("font-weight", (esSeleccionada || esDeItinerario) ? "700" : "500");
    txt.setAttribute("pointer-events", "none");
    txt.textContent = `${{a.distancia_km}}km · ${{a.tiempo_efectivo_h.toFixed(2)}}h`;

    gEdge.appendChild(pathVis);
    gEdge.appendChild(pathHit);
    gEdge.appendChild(txt);
    layerEdges.appendChild(gEdge);
  }});

  // Dibujar Nodos (Vértices)
  nodos.forEach(n => {{
    const pos = agyPosiciones[n.id];
    if (!pos) return;

    const esSeleccionado = agySeleccion && agySeleccion.tipo === "nodo" && agySeleccion.id === n.id;
    const esDeItin = nodosItinerario.has(n.id);
    const color = AGY_COLORES_NODO[n.tipo] || "#64748b";

    const gNode = document.createElementNS("http://www.w3.org/2000/svg", "g");
    gNode.setAttribute("class", "agy-node-group");
    gNode.setAttribute("transform", `translate(${{pos.x}}, ${{pos.y}})`);

    // Halo exterior para zonas D (deterioro) y H (hábitat restringido) o nodo seleccionado
    if (n.tipo === "D" || n.tipo === "H" || esSeleccionado || esDeItin) {{
      const halo = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      halo.setAttribute("r", esSeleccionado ? "24" : "21");
      halo.setAttribute("fill", n.tipo === "D" ? "rgba(244,67,54,0.18)" : (n.tipo === "H" ? "rgba(76,175,80,0.18)" : "rgba(250,204,21,0.2)"));
      halo.setAttribute("stroke", esSeleccionado ? "#facc15" : (n.tipo === "H" ? "#ef4444" : color));
      halo.setAttribute("stroke-width", "1.8");
      if (n.tipo === "H") halo.setAttribute("stroke-dasharray", "4,3");
      gNode.appendChild(halo);
    }}

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", "agy-node-circle");
    circle.setAttribute("r", "14");
    circle.setAttribute("fill", color);
    circle.setAttribute("stroke", esSeleccionado ? "#facc15" : "#f8fafc");
    circle.setAttribute("stroke-width", esSeleccionado ? "3.5" : "2");
    gNode.appendChild(circle);

    // Código del vértice dentro del círculo
    const lblId = document.createElementNS("http://www.w3.org/2000/svg", "text");
    lblId.setAttribute("text-anchor", "middle");
    lblId.setAttribute("dy", "3.5");
    lblId.setAttribute("fill", "#ffffff");
    lblId.setAttribute("font-size", "9.5px");
    lblId.setAttribute("font-weight", "700");
    lblId.setAttribute("pointer-events", "none");
    lblId.textContent = n.id;
    gNode.appendChild(lblId);

    // Nombre debajo del círculo
    const lblName = document.createElementNS("http://www.w3.org/2000/svg", "text");
    lblName.setAttribute("text-anchor", "middle");
    lblName.setAttribute("dy", "27");
    lblName.setAttribute("fill", "#f1f5f9");
    lblName.setAttribute("font-size", "10.5px");
    lblName.setAttribute("font-weight", "600");
    lblName.setAttribute("pointer-events", "none");
    lblName.textContent = n.nombre;
    gNode.appendChild(lblName);

    // Eventos de clic (inspeccionar) y arrastre (mover nodo)
    gNode.addEventListener("mousedown", (ev) => {{
      ev.stopPropagation();
      agyDraggingNode = n.id;
    }});
    gNode.addEventListener("click", (ev) => {{
      ev.stopPropagation();
      agyInspeccionarNodo(n);
    }});

    layerNodes.appendChild(gNode);
  }});
}}

// Mostrar el resumen general del subgrafo cuando aún no se ha hecho clic en un nodo o arista
function agyMostrarResumenSubgrafo() {{
  agySeleccion = null;
  const {{ nodos, aristas }} = agySubgrafoActual();
  document.getElementById("agy-insp-title").textContent = "🔎 Estructura del Grafo en Memoria";
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `${{nodos.length}} V · ${{aristas.length}} E`;
  badge.style.background = "#1e293b";
  badge.style.color = "#38bdf8";

  // Construir vista de la Lista de Adyacencia real (_adyacencia)
  const adjLines = nodos.map(n => {{
    const salientes = aristas.filter(a => a.origen === n.id).map(a => `${{a.destino}}(w=${{a.peso_combinado.toFixed(3)}})`).join(", ");
    return `_adyacencia["${{n.id}}"] -> [${{salientes}}]`;
  }}).join("\\n");

  const chipsNodos = nodos.map(n =>
    `<span class="agy-chip" onclick="agySeleccionarElementoRapido('N:${{n.id}}')" style="border-left:3px solid ${{AGY_COLORES_NODO[n.tipo]}}">${{n.id}} · ${{n.nombre}}</span>`
  ).join("");

  const chipsAristas = aristas.slice(0, 18).map(a =>
    `<span class="agy-chip" onclick="agySeleccionarElementoRapido('E:${{a.id}}')">${{a.origen}} → ${{a.destino}} (${{a.tipo_via}})</span>`
  ).join("");

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      👆 <b>Haz clic sobre cualquier vértice (círculo) o arista (flecha)</b> en el lienzo para inspeccionar todas las variables y propiedades que almacena ese objeto en Python.
    </div>

    <div class="agy-prop-group">
      <h4>📐 Resumen Matemático de Esta Vista</h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Vértices |V|</span><span class="agy-prop-val">${{nodos.length}} nodos</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Aristas dirigidas |E|</span><span class="agy-prop-val">${{aristas.length}} arcos</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Representación</span><span class="agy-prop-val">Lista de Adyacencia O(|V|+|E|)</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Fórmula de peso w_ij</span><span class="agy-prop-val">6 factores (22 variables)</span></div>
    </div>

    <div class="agy-prop-group">
      <h4>📍 Selección Rápida de Vértices</h4>
      <div class="agy-chip-list">${{chipsNodos}}</div>
    </div>

    <div class="agy-prop-group">
      <h4>🔗 Selección Rápida de Aristas</h4>
      <div class="agy-chip-list">${{chipsAristas}}</div>
    </div>

    <div class="agy-prop-group">
      <h4>💻 Lista de Adyacencia en Memoria (_adyacencia)</h4>
      <div class="agy-code-block">${{adjLines}}</div>
    </div>
  `;
}}

// Inspeccionar en detalle un Vértice (Nodo) al hacer clic sobre él
function agyInspeccionarNodo(n) {{
  agySeleccion = {{ tipo: "nodo", id: n.id }};
  agyRenderizarGrafo();

  document.getElementById("agy-insp-title").textContent = `📍 Vértice ${{n.id}}: ${{n.nombre}}`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `Tipo ${{n.tipo}} · ${{n.tipo_descripcion}}`;
  badge.style.background = AGY_COLORES_NODO[n.tipo] || "#334155";
  badge.style.color = "#fff";

  const aristasSalientes = AGY_DATA.aristas.filter(a => a.origen === n.id);
  const aristasEntrantes = AGY_DATA.aristas.filter(a => a.destino === n.id);

  const chipsSalientes = aristasSalientes.length
    ? aristasSalientes.map(a => `<span class="agy-chip" onclick="agySeleccionarElementoRapido('E:${{a.id}}')">→ ${{a.destino}} (${{a.tipo_via}}, ${{a.distancia_km}} km, w=${{a.peso_combinado.toFixed(3)}})</span>`).join("")
    : `<span style="color:#64748b">Ninguna (grado de salida = 0)</span>`;

  const chipsEntrantes = aristasEntrantes.length
    ? aristasEntrantes.map(a => `<span class="agy-chip" onclick="agySeleccionarElementoRapido('E:${{a.id}}')">← ${{a.origen}} (${{a.tipo_via}}, ${{a.distancia_km}} km)</span>`).join("")
    : `<span style="color:#64748b">Ninguna (grado de entrada = 0)</span>`;

  const pythonObj = JSON.stringify({{
    id: n.id,
    nombre: n.nombre,
    tipo: n.tipo,
    latitud: n.latitud,
    longitud: n.longitud,
    produccion_kg_dia: n.produccion_kg_dia,
    capacidad_frio_kg: n.capacidad_frio_kg,
    costo_fijo: n.costo_fijo,
    tiene_energia: n.tiene_energia,
    tiene_agua: n.tiene_agua,
    tiene_muelle: n.tiene_muelle,
    restriccion_ambiental: n.restriccion_ambiental,
    metadata: n.metadata
  }}, null, 2);

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <b>⚡ Efecto real de este vértice en el grafo:</b><br>
      ${{n.efecto_vertice}}
    </div>

    <div class="agy-prop-group">
      <h4>1. Atributos Geográficos y de Identidad (Nodo)</h4>
      <div class="agy-prop-row"><span class="agy-prop-key">nodo.id</span><span class="agy-prop-val">"${{n.id}}"</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">nodo.nombre</span><span class="agy-prop-val">${{n.nombre}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">nodo.tipo</span><span class="agy-prop-val">${{n.tipo}} (${{n.tipo_descripcion}})</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">nodo.coordenadas</span><span class="agy-prop-val">(${{n.latitud.toFixed(4)}}, ${{n.longitud.toFixed(4)}})</span></div>
    </div>

    <div class="agy-prop-group">
      <h4>2. Capacidades, Producción y Costos</h4>
      <div class="agy-prop-row"><span class="agy-prop-key">produccion_kg_dia</span><span class="agy-prop-val">${{n.produccion_kg_dia.toFixed(2)}} kg/día</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">capacidad_frio_kg</span><span class="agy-prop-val">${{n.capacidad_frio_kg.toLocaleString()}} kg</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">costo_fijo</span><span class="agy-prop-val">$${{n.costo_fijo.toLocaleString()}} COP/mes</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">demanda_kg_dia</span><span class="agy-prop-val">${{n.metadata.demanda_kg_dia || 0}} kg/día</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">rio_asociado</span><span class="agy-prop-val">${{n.metadata.rio || "—"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">num_pescadores</span><span class="agy-prop-val">${{n.metadata.pescadores || "—"}}</span></div>
    </div>

    <div class="agy-prop-group">
      <h4>3. Banderas de Infraestructura y Restricciones</h4>
      <div class="agy-prop-row"><span class="agy-prop-key">tiene_energia</span><span class="agy-prop-val">${{n.tiene_energia ? "✅ True" : "❌ False"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">tiene_agua</span><span class="agy-prop-val">${{n.tiene_agua ? "✅ True" : "❌ False"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">tiene_muelle</span><span class="agy-prop-val">${{n.tiene_muelle ? "✅ True" : "❌ False"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">restriccion_ambiental</span><span class="agy-prop-val">${{n.restriccion_ambiental ? "🚫 True (Hábitat H)" : "False"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">es_candidato_frio (FLP)</span><span class="agy-prop-val">${{n.es_candidato_frio ? "✅ Sí es candidato" : "❌ No"}}</span></div>
    </div>

    <div class="agy-prop-group">
      <h4>4. Topología en la Lista de Adyacencia</h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Grado de Entrada (deg⁻)</span><span class="agy-prop-val">${{n.grado_entrada}} aristas entrantes</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Grado de Salida (deg⁺)</span><span class="agy-prop-val">${{n.grado_salida}} aristas salientes</span></div>
      <div style="margin-top:8px;color:#94a3b8;font-size:11px;">Aristas que salen de ${{n.id}}:</div>
      <div class="agy-chip-list">${{chipsSalientes}}</div>
      <div style="margin-top:8px;color:#94a3b8;font-size:11px;">Aristas que llegan a ${{n.id}}:</div>
      <div class="agy-chip-list">${{chipsEntrantes}}</div>
    </div>

    <div class="agy-prop-group">
      <h4>5. Objeto Nodo Serializado en Memoria</h4>
      <div class="agy-code-block">${{pythonObj}}</div>
    </div>
  `;
}}

// Inspeccionar en detalle una Arista (sus 22 variables y el desglose de su fórmula multicriterio)
function agyInspeccionarArista(a) {{
  agySeleccion = {{ tipo: "arista", id: a.id }};
  agyRenderizarGrafo();

  const nO = AGY_DATA.nodos.find(x => x.id === a.origen);
  const nD = AGY_DATA.nodos.find(x => x.id === a.destino);

  document.getElementById("agy-insp-title").textContent = `🔗 Arista #${{String(a.idx).padStart(2, '0')}}: ${{a.origen}} → ${{a.destino}}`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `${{a.tipo_via}} (${{a.estado_via}})`;
  badge.style.background = a.tipo_via === "fluvial" ? "#0284c7" : (a.tipo_via === "acceso_local" ? "#475569" : "#ea580c");
  badge.style.color = "#fff";

  const w = a.peso_combinado || 0.0001;
  const d = a.desglose_peso;
  const pct = (val) => Math.min(100, Math.round((val / w) * 100));

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <b>${{nO ? nO.nombre : a.origen}} (${{a.origen}}) ➔ ${{nD ? nD.nombre : a.destino}} (${{a.destino}})</b><br>
      Geometría real almacenada: <b>${{a.num_puntos_geometria}} coordenadas GPS</b> (${{a.tipo_via === "fluvial" ? "Cauce hidrográfico rios.shp" : "Trazado vial OSRM"}}).
      <div class="agy-chip-list" style="margin-top:6px;">
        <span class="agy-chip" onclick="agySeleccionarElementoRapido('N:${{a.origen}}')">📍 Ver origen ${{a.origen}}</span>
        <span class="agy-chip" onclick="agySeleccionarElementoRapido('N:${{a.destino}}')">🏁 Ver destino ${{a.destino}}</span>
      </div>
    </div>

    <div class="agy-prop-group">
      <h4>⚖️ Desglose de la Función Multicriterio (w_ij = ${{a.peso_combinado.toFixed(4)}})</h4>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span>α·Tiempo efectivo (0.25)</span><span>+${{d.t_term.toFixed(4)}} (${{pct(d.t_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.t_term)}}%"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span>β·Costo total c/peajes (0.20)</span><span>+${{d.c_term.toFixed(4)}} (${{pct(d.c_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.c_term)}}%"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span>γ·Riesgo combinado (0.20)</span><span>+${{d.r_term.toFixed(4)}} (${{pct(d.r_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.r_term)}}%"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span>δ·Pérdida esperada (0.10)</span><span>+${{d.p_term.toFixed(4)}} (${{pct(d.p_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.p_term)}}%"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span>ε·Incertidumbre 1-conf (0.10)</span><span>+${{d.conf_term.toFixed(4)}} (${{pct(d.conf_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.conf_term)}}%"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span>ζ·Prob. de retraso (0.05)</span><span>+${{d.ret_term.toFixed(4)}} (${{pct(d.ret_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.ret_term)}}%"></div></div>
      </div>
    </div>

    <div class="agy-prop-group">
      <h4>1. Variables de Distancia, Tiempo y Flujo</h4>
      <div class="agy-prop-row"><span class="agy-prop-key">distancia_km</span><span class="agy-prop-val">${{a.distancia_km.toFixed(2)}} km</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">tiempo_h (base)</span><span class="agy-prop-val">${{a.tiempo_h.toFixed(2)}} h</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">flujo_vehicular (factor)</span><span class="agy-prop-val">${{a.flujo_vehicular}} (×${{a.factor_flujo}})</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">tiempo_espera_h</span><span class="agy-prop-val">${{a.tiempo_espera_h.toFixed(2)}} h</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">tiempo_efectivo_h</span><span class="agy-prop-val">${{a.tiempo_efectivo_h.toFixed(3)}} h</span></div>
    </div>

    <div class="agy-prop-group">
      <h4>2. Variables Financieras y Peajes</h4>
      <div class="agy-prop-row"><span class="agy-prop-key">costo_cop (flete)</span><span class="agy-prop-val">$${{a.costo_cop.toLocaleString()}} COP</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">peajes (cantidad)</span><span class="agy-prop-val">${{a.peajes}} caseta(s)</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">costo_peajes_cop</span><span class="agy-prop-val">$${{a.costo_peajes_cop.toLocaleString()}} COP</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">costo_total_cop</span><span class="agy-prop-val">$${{a.costo_total_cop.toLocaleString()}} COP</span></div>
    </div>

    <div class="agy-prop-group">
      <h4>3. Variables de Riesgo, Vía y Deterioro</h4>
      <div class="agy-prop-row"><span class="agy-prop-key">tipo_via</span><span class="agy-prop-val">${{a.tipo_via}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">estado_via (factor)</span><span class="agy-prop-val">${{a.estado_via}} (×${{a.factor_estado_via}})</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">confiabilidad_via</span><span class="agy-prop-val">${{a.confiabilidad_via.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">prob_retraso</span><span class="agy-prop-val">${{a.prob_retraso.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">riesgo_termico</span><span class="agy-prop-val">${{a.riesgo_termico.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">riesgo_accidente</span><span class="agy-prop-val">${{a.riesgo_accidente.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">riesgo_derrumbe</span><span class="agy-prop-val">${{a.riesgo_derrumbe.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">riesgo_combinado</span><span class="agy-prop-val">${{a.riesgo_combinado.toFixed(4)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">perdida_esperada (L_ij)</span><span class="agy-prop-val">${{a.perdida_esperada.toFixed(2)}}%</span></div>
    </div>

    <div class="agy-prop-group">
      <h4>4. Restricciones Operativas</h4>
      <div class="agy-prop-row"><span class="agy-prop-key">capacidad_kg</span><span class="agy-prop-val">${{a.capacidad_kg.toLocaleString()}} kg</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">restriccion_peso_ton</span><span class="agy-prop-val">${{a.restriccion_peso_ton}} ton</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">horario_restriccion</span><span class="agy-prop-val">${{a.horario_restriccion}}</span></div>
    </div>
  `;
}}

function agyInspeccionarItinerario(it) {{
  document.getElementById("agy-insp-title").textContent = `🐟 Itinerario ${{it.origen_id}} → ${{it.destino_id}}`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `${{it.distancia_total_km}} km · ${{it.tiempo_total_h}} h`;
  badge.style.background = "#ca8a04";
  badge.style.color = "#fff";

  const filasTramos = (it.tramos || []).map((t, i) => `
    <div class="agy-prop-row" style="cursor:pointer" onclick="agySeleccionarElementoRapido('E:${{t.origen}}->${{t.destino}}')">
      <span class="agy-prop-key">${{i+1}}. ${{t.origen}} → ${{t.destino}} (${{t.tipo_via}})</span>
      <span class="agy-prop-val">${{t.distancia_km}} km · ${{t.tiempo_h}} h ${{t.efecto ? '· ' + t.efecto : ''}}</span>
    </div>
  `).join("");

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <b>Secuencia Multimodal:</b> ${{it.ruta.join(" ➔ ")}}<br>
      ${{it.nodos_restringidos_evitados.length ? `🌿 <b>Hábitats protegidos evitados:</b> ${{it.nodos_restringidos_evitados.join(", ")}}` : "✅ Ruta limpia de restricciones ambientales."}}
    </div>
    <div class="agy-prop-group">
      <h4>📊 Métricas Consolidadas del Recorrido</h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Acopio asignado</span><span class="agy-prop-val">${{it.acopio_id || " Directo urbano"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Centro de Frío</span><span class="agy-prop-val">${{it.centro_frio_id}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Distancia Total</span><span class="agy-prop-val">${{it.distancia_total_km}} km</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Tiempo Total</span><span class="agy-prop-val">${{it.tiempo_total_h}} h</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Carga Inicial → Final</span><span class="agy-prop-val">${{it.kg_inicial}} kg → ${{it.kg_final}} kg</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Pérdida Térmica</span><span class="agy-prop-val">${{it.perdida_pct}}%</span></div>
    </div>
    <div class="agy-prop-group">
      <h4>🛣️ Tramos del Itinerario (clic para inspeccionar arista)</h4>
      ${{filasTramos}}
    </div>
  `;
}}

// Interacción de arrastre de nodos, paneo y zoom sobre el SVG
(function initEventosCanvas() {{
  const svg = document.getElementById("agy-graph-svg");

  svg.addEventListener("mousedown", (ev) => {{
    if (agyDraggingNode) return;
    agyPanning = true;
    agyPanStart = {{ x: ev.clientX - agyTransform.x, y: ev.clientY - agyTransform.y }};
  }});

  window.addEventListener("mousemove", (ev) => {{
    if (agyDraggingNode) {{
      const rect = svg.getBoundingClientRect();
      const x = (ev.clientX - rect.left - agyTransform.x) / agyTransform.k;
      const y = (ev.clientY - rect.top - agyTransform.y) / agyTransform.k;
      agyPosiciones[agyDraggingNode] = {{ x, y }};
      agyRenderizarGrafo();
    }} else if (agyPanning) {{
      agyTransform.x = ev.clientX - agyPanStart.x;
      agyTransform.y = ev.clientY - agyPanStart.y;
      agyActualizarTransform();
    }}
  }});

  window.addEventListener("mouseup", () => {{
    agyDraggingNode = null;
    agyPanning = false;
  }});

  svg.addEventListener("wheel", (ev) => {{
    ev.preventDefault();
    const factor = ev.deltaY < 0 ? 1.12 : 0.89;
    const nuevoK = Math.min(3.5, Math.max(0.45, agyTransform.k * factor));
    agyTransform.k = nuevoK;
    agyActualizarTransform();
  }}, {{ passive: false }});

  // Navegación con flechas del teclado (← Anterior / Siguiente →)
  window.addEventListener("keydown", (ev) => {{
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    if (ev.key === "ArrowRight") {{
      agyCambiarSlide(agySlideActual + 1);
    }} else if (ev.key === "ArrowLeft") {{
      agyCambiarSlide(agySlideActual - 1);
    }}
  }});
}})();
</script>
"""
