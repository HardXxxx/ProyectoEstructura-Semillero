"""
graph_slides.py — Plataforma Interactiva de Visualización Geográfica y Teoría de Grafos
para la Red Logística Multimodal de Pesca Artesanal (Puerto Gaitán, Meta → Bogotá).

Genera:
  1. `data/output/presentacion_red_logistica.html` — Aplicación web interactiva 100% autónoma
     y funcional sin conexión (file://, cero CDNs externos), estructurada en 4 diapositivas:
       • Diapositiva 1: Mapa Geográfico SIG Vectorial (ríos reales rios.shp, rutas viales OSRM
         en caché, 25 vértices del modelo, 49 puntos de campo, retícula WGS84, control de capas).
       • Diapositiva 2: Grafo Vial (Subgrafo exclusivo terrestre + acceso_local, 31 aristas,
         jerarquía vial y estado de superficie).
       • Diapositiva 3: Grafo Fluvial (Subgrafo de 10 conexiones navegables frente a la red
         hidrográfica pasiva de 12,413 vértices de rios.shp).
       • Diapositiva 4: Grafo Mixto Multimodal (Red completa de 25 vértices y 41 aristas,
         nodos de transferencia modal, itinerarios óptimos y comparador de escenarios reales).
  2. `generar_html_slides(...)` — Bloque integrable en `folium_map.py`.

Diseñado bajo los estándares de:
  • Emil Kowalski (`animate`, `animation-performance`, `improve-animations`, `review-animations`)
  • Impeccable Design (`impeccable-design` — grilla 4/8px, escala tipográfica, proporción 60-30-10)
  • Taste (`taste.md` — dirección de arte técnica SIG, anti-slop, números tabulares)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..models.grafo_logistico import GrafoLogistico
from ..algorithms.dijkstra import dijkstra
from ..algorithms.bellman_ford import bellman_ford
from ..optimization.facility_location import resolver_localizacion
from ..optimization.scenario_evaluator import EvaluadorEscenarios
from ..optimization.itinerarios import calcular_itinerarios
from ..utils.constants import (
    ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA,
    TIEMPO_MAX_SIN_FRIO_H, TIEMPO_TRANSBORDO_MUELLE_H, FACTOR_DETERIORO_ZONA_D,
)
from ..utils.data_loader import DATA_DIR, OUTPUT_DIR

GEOJSON_DIR = DATA_DIR / "geojson"


def _cargar_puntos_campo_reales() -> List[Dict[str, Any]]:
    """Carga los 49 puntos de campo reales desde data/geojson/puntos_campo.geojson."""
    ruta = GEOJSON_DIR / "puntos_campo.geojson"
    if not ruta.exists():
        return []
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            gj = json.load(f)
        puntos = []
        for feat in gj.get("features", []):
            coords = feat.get("geometry", {}).get("coordinates", [])
            props = feat.get("properties", {})
            if len(coords) >= 2:
                puntos.append({
                    "id": props.get("id", ""),
                    "categoria": props.get("categoria", ""),
                    "tipo": props.get("tipo", ""),
                    "categoria_texto": props.get("categoria_texto", ""),
                    "lat_dms": props.get("lat_dms", ""),
                    "lon_dms": props.get("lon_dms", ""),
                    "longitud": round(float(coords[0]), 6),
                    "latitud": round(float(coords[1]), 6),
                    "title": props.get("title", ""),
                })
        return puntos
    except Exception:
        return []


def construir_datos_slides(
    grafo: GrafoLogistico,
    geo_cache: Optional[Dict[str, list]] = None,
    itinerarios: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Extrae y serializa los datos reales del repositorio (nodos, aristas, geometrías OSRM,
    red hidrográfica rios.shp, 49 puntos de campo, itinerarios y resultados de algoritmos)
    sin inventar valores ni alterar el modelo matemático.
    """
    if not geo_cache:
        ruta_cache = OUTPUT_DIR / "rutas_cache.json"
        if ruta_cache.exists():
            try:
                with open(ruta_cache, "r", encoding="utf-8") as f:
                    geo_cache = json.load(f)
            except Exception:
                geo_cache = {}
        else:
            geo_cache = {}

    # 1. Grados de entrada/salida y modos de transporte por vértice
    grado_entrada: Dict[str, int] = {n.id: 0 for n in grafo.nodos}
    grado_salida: Dict[str, int] = {n.id: 0 for n in grafo.nodos}
    vecinos_salida: Dict[str, List[str]] = {n.id: [] for n in grafo.nodos}
    vecinos_entrada: Dict[str, List[str]] = {n.id: [] for n in grafo.nodos}
    modos_incidentes: Dict[str, set] = {n.id: set() for n in grafo.nodos}

    for ar in grafo.aristas:
        grado_salida[ar.origen] = grado_salida.get(ar.origen, 0) + 1
        grado_entrada[ar.destino] = grado_entrada.get(ar.destino, 0) + 1
        vecinos_salida.setdefault(ar.origen, []).append(ar.destino)
        vecinos_entrada.setdefault(ar.destino, []).append(ar.origen)
        modos_incidentes.setdefault(ar.origen, set()).add(ar.tipo_via)
        modos_incidentes.setdefault(ar.destino, set()).add(ar.tipo_via)

    # 2. Serializar los 25 Nodos reales
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
        modos = sorted(list(modos_incidentes.get(n.id, set())))
        d["modos_conectados"] = modos
        d["es_multimodal"] = ("fluvial" in modos and len(modos) > 1) or n.tipo == "M"
        d["metadata"] = n.metadata or {}

        if n.tipo == "M":
            d["efecto_vertice"] = (
                f"Nodo de transferencia modal (fluvial → terrestre): añade +{TIEMPO_TRANSBORDO_MUELLE_H:.2f} h "
                "al tiempo acumulado sin frío y contabiliza +1 operación de transbordo."
            )
        elif n.tipo == "D":
            d["efecto_vertice"] = (
                f"Zona crítica de deterioro térmico: multiplica por ×{FACTOR_DETERIORO_ZONA_D} "
                "la tasa de pérdida del tramo entrante por exposición superficial sin sombra."
            )
        elif n.tipo == "H":
            d["efecto_vertice"] = (
                "Hábitat protegido (restriccion_ambiental=True): bloqueado como vértice "
                "intermedio en Dijkstra, Bellman-Ford y Floyd-Warshall cuando evitar_restringidos=True."
            )
        elif n.tipo == "F":
            d["efecto_vertice"] = (
                f"Planta de refrigeración ({n.capacidad_frio_kg:,.0f} kg, costo fijo ${n.costo_fijo:,.0f} COP): "
                "abate la temperatura del lote y reinicia tiempo_sin_frio = 0.0 h."
            )
        elif n.tipo == "A":
            d["efecto_vertice"] = (
                f"Centro de acopio obligatorio (ADR-06, capacidad {n.capacidad_frio_kg:,.0f} kg): "
                + ("dispone de energía eléctrica confiable y pre-frío."
                   if n.tiene_energia else "sin energía eléctrica confiable (excluido como candidato FLP).")
            )
        elif n.tipo == "P":
            d["efecto_vertice"] = (
                f"Punto de captura artesanal ({n.metadata.get('rio', 'Cuenca Meta')}): "
                f"produce {n.produccion_kg_dia:.2f} kg/día con {n.metadata.get('pescadores', 0)} pescadores."
            )
        else:
            d["efecto_vertice"] = (
                f"Mercado comercial de destino final (demanda diaria: {n.metadata.get('demanda_kg_dia', 0)} kg/día)."
            )
        nodos_data.append(d)

    # 3. Serializar las 41 Aristas reales con sus 22 atributos y geometría GPS real
    aristas_data = []
    for idx, ar in enumerate(grafo.aristas, start=1):
        ad = ar.to_dict()
        ad["idx"] = idx
        ad["id"] = f"{ar.origen}->{ar.destino}"
        ad["factor_estado_via"] = ar.factor_estado_via
        ad["factor_flujo"] = ar.factor_flujo
        vel_ef = ar.distancia_km / ar.tiempo_efectivo_h if ar.tiempo_efectivo_h > 0 else 0.0
        ad["velocidad_efectiva_kmh"] = round(vel_ef, 2)

        clave_geo = f"{ar.origen}-{ar.destino}"
        pts_geo = geo_cache.get(clave_geo) or []
        ad["num_puntos_geometria"] = len(pts_geo)
        if len(pts_geo) > 75:
            paso = max(1, len(pts_geo) // 70)
            muestra = [
                [round(float(p[0]), 5), round(float(p[1]), 5)]
                for p in pts_geo[::paso]
            ]
            ultimo = [round(float(pts_geo[-1][0]), 5), round(float(pts_geo[-1][1]), 5)]
            if muestra[-1] != ultimo:
                muestra.append(ultimo)
            ad["geometria_muestra"] = muestra
        elif len(pts_geo) >= 2:
            ad["geometria_muestra"] = [
                [round(float(p[0]), 5), round(float(p[1]), 5)] for p in pts_geo
            ]
        else:
            n_orig = grafo.obtener_nodo(ar.origen)
            n_dest = grafo.obtener_nodo(ar.destino)
            if n_orig and n_dest:
                ad["geometria_muestra"] = [
                    [round(n_orig.latitud, 5), round(n_orig.longitud, 5)],
                    [round(n_dest.latitud, 5), round(n_dest.longitud, 5)],
                ]
            else:
                ad["geometria_muestra"] = []

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

    # 4. Calcular Itinerarios reales (Bogotá V_BOG1 y Local V_PG1)
    def _serializar_lista_itinerarios(lista_it: list) -> List[Dict[str, Any]]:
        out = []
        for it in lista_it:
            out.append({
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
                        "distancia_km": round(t.distancia_km, 2),
                        "tiempo_h": round(t.tiempo_h, 2),
                        "tipo_nodo_destino": t.tipo_nodo_destino,
                        "efecto": t.efecto,
                    }
                    for t in getattr(it, "tramos", [])
                ],
            })
        return out

    itinerarios_bog = list(itinerarios) if itinerarios else calcular_itinerarios(grafo, "V_BOG1")
    itinerarios_pg = calcular_itinerarios(grafo, "V_PG1")

    # 5. Resultados reales de FLP, Bellman-Ford y Evaluador de Escenarios
    flp_data = {}
    try:
        res_flp = resolver_localizacion(grafo, metodo="greedy", tiempo_max_h=TIEMPO_MAX_SIN_FRIO_H)
        flp_data = {
            "metodo": "greedy",
            "factible": bool(res_flp.cobertura >= 99.9),
            "centros_abiertos": list(res_flp.centros_abiertos),
            "asignaciones": dict(res_flp.asignaciones),
            "costo_total_fijo": round(res_flp.costo_total, 0),
            "tiempo_max_acceso_h": round(res_flp.tiempo_max_sin_frio, 3),
            "cobertura_pct": round(res_flp.cobertura, 1),
        }
    except Exception:
        flp_data = {}

    escenarios_pg = []
    escenarios_bog = []
    try:
        ev = EvaluadorEscenarios(grafo, tiempo_limite_h=TIEMPO_MAX_SIN_FRIO_H)
        defs_esc = [
            {"nombre": "Sin cadena de frío", "centros": []},
            {"nombre": "Solo F01 (Alto Neblinas)", "centros": ["F01"]},
            {"nombre": "Solo F02 (Puerto Gaitán)", "centros": ["F02"]},
            {"nombre": "Solo F03 (Puerto López)", "centros": ["F03"]},
            {"nombre": "3 centros combinados (F01+F02+F03)", "centros": ["F01", "F02", "F03"]},
        ]
        for m in ev.comparar_escenarios(defs_esc, "V_PG1"):
            escenarios_pg.append({
                "nombre": m.nombre,
                "num_centros": m.num_centros,
                "centros_ids": m.centros_ids,
                "pct_perdida": round(m.pct_perdida, 2),
                "kg_perdido": round(m.kg_perdido, 1),
                "costo_total_cop": round(m.costo_total_cop, 0),
                "costo_por_kg_cop": round(m.costo_por_kg_cop, 0),
                "cumplimiento_termico_pct": round(m.cumplimiento_termico_pct, 1),
            })
        for m in ev.comparar_escenarios(defs_esc, "V_BOG1"):
            escenarios_bog.append({
                "nombre": m.nombre,
                "num_centros": m.num_centros,
                "centros_ids": m.centros_ids,
                "pct_perdida": round(m.pct_perdida, 2),
                "kg_perdido": round(m.kg_perdido, 1),
                "costo_total_cop": round(m.costo_total_cop, 0),
                "costo_por_kg_cop": round(m.costo_por_kg_cop, 0),
                "cumplimiento_termico_pct": round(m.cumplimiento_termico_pct, 1),
            })
    except Exception:
        pass

    # 6. Red Fluvial Topológica (rios.shp) y 49 Puntos de Campo (puntos_campo.geojson)
    rios_lineas = []
    stats_rio = {"vertices": 12413, "aristas_dirigidas": 24790, "celdas_rejilla": 447}
    try:
        from .river_network import obtener_capa_rios_geojson
        capa_rios = obtener_capa_rios_geojson(tolerancia=0.0018)
        if capa_rios and capa_rios.get("features"):
            for feat in capa_rios["features"]:
                coords = feat.get("geometry", {}).get("coordinates", [])
                if len(coords) >= 2:
                    rios_lineas.append([
                        [round(float(c[0]), 4), round(float(c[1]), 4)]
                        for c in coords
                    ])
    except Exception:
        pass

    puntos_campo = _cargar_puntos_campo_reales()

    return {
        "nodos": nodos_data,
        "aristas": aristas_data,
        "itinerarios": _serializar_lista_itinerarios(itinerarios_bog),
        "itinerarios_locales": _serializar_lista_itinerarios(itinerarios_pg),
        "flp": flp_data,
        "escenarios_pg": escenarios_pg,
        "escenarios_bog": escenarios_bog,
        "rios_lineas": rios_lineas,
        "stats_rio": stats_rio,
        "puntos_campo": puntos_campo,
        "constantes": {
            "ALPHA": ALPHA,
            "BETA": BETA,
            "GAMMA": GAMMA,
            "DELTA": DELTA,
            "EPSILON": EPSILON,
            "ZETA": ZETA,
            "TIEMPO_MAX_SIN_FRIO_H": TIEMPO_MAX_SIN_FRIO_H,
            "TIEMPO_TRANSBORDO_MUELLE_H": TIEMPO_TRANSBORDO_MUELLE_H,
            "FACTOR_DETERIORO_ZONA_D": FACTOR_DETERIORO_ZONA_D,
        },
    }


def _construir_bloque_ui_interactiva(json_payload: str, modo_autonomo: bool = True) -> str:
    """
    Construye el sistema completo de estilos, barra de navegación global,
    las 4 vistas (Mapa Geográfico SIG Vectorial Offline, Grafo Vial, Grafo Fluvial,
    Grafo Mixto Multimodal), el Inspector Lateral y el motor interactivo en JS Vanilla.
    """
    clase_slide0 = "visible" if modo_autonomo else ""
    estilo_puente_folium = "" if modo_autonomo else """
  /* En modo integrado sobre Folium, la Diapositiva 1 puede alternar entre el SIG Vectorial y el mapa Leaflet */
  #agy-gis-deck.folium-transparent {
    display: none !important;
  }
"""
    return f"""
<style>
  :root {{
    /* Sistema de Superficies por Capas — Regla 60-30-10 & Anti-Slop (taste.md) */
    --bg-canvas: #090b0e;            /* 60% Superficie dominante mate */
    --bg-gis-water: #0c1219;         /* Fondo cartográfico batimétrico */
    --bg-panel: #11141a;             /* 30% Estructura lateral y cabeceras */
    --bg-elevated: #171b22;          /* Tarjetas, selectores y controles */
    --bg-hover: #1e242d;             /* Hover sutil */
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-strong: rgba(255, 255, 255, 0.16);

    /* Jerarquía Tipográfica y Contraste WCAG AA (impeccable-design) */
    --text-primary: #f3f4f6;         /* 95% luminosidad */
    --text-secondary: #9ca3af;       /* 65% luminosidad */
    --text-muted: #6b7280;

    /* 10% Paleta Semántica de Ingeniería Logística */
    --col-pesca: #38bdf8;            /* P: Captura artesanal */
    --col-muelle: #a855f7;           /* M: Transbordo modal */
    --col-acopio: #f59e0b;           /* A: Centro de acopio */
    --col-frio: #06b6d4;             /* F: Planta de refrigeración */
    --col-deterioro: #ef4444;        /* D: Zona crítica de deterioro */
    --col-habitat: #10b981;          /* H: Hábitat protegido */
    --col-venta: #ec4899;            /* V: Mercado destino */
    --col-highlight: #facc15;        /* Ruta óptima / selección */
    --accent: #38bdf8;
    --accent-subtle: rgba(56, 189, 248, 0.12);
    --accent-border: rgba(56, 189, 248, 0.38);

    /* Curvas Exactas Emil Kowalski (review-animations/STANDARDS.md) */
    --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
    --ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
    --ease-drawer: cubic-bezier(0.32, 0.72, 0, 1);

    --font-sans: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'Cascadia Code', Consolas, 'Courier New', monospace;
  }}

  *, *::before, *::after {{
    box-sizing: border-box;
  }}

  /* Números tabulares obligatorios en toda métrica, coordenada, peso y contador */
  .tabular, .agy-kpi-val, .agy-prop-val, .agy-bar-label, .agy-code-block,
  #agy-slide-counter, #agy-coords-hud, .agy-table td, .agy-table th {{
    font-variant-numeric: tabular-nums;
  }}

  /* ═══════════════════════════════════════════════════════════════
     BARRA GLOBAL DE NAVEGACIÓN (Altura 56px = 7 × 8px)
     ═══════════════════════════════════════════════════════════════ */
  #agy-slide-nav {{
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 56px;
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: rgba(17, 20, 26, 0.96);
    backdrop-filter: blur(10px);
    color: var(--text-primary);
    padding: 0 16px;
    border-bottom: 1px solid var(--border-subtle);
    font-family: var(--font-sans);
    user-select: none;
  }}

  .agy-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }}
  .agy-brand-mark {{
    width: 28px;
    height: 28px;
    border-radius: 6px;
    background: var(--accent-subtle);
    border: 1px solid var(--accent-border);
    color: var(--accent);
    display: grid;
    place-items: center;
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 700;
    flex-shrink: 0;
  }}
  .agy-brand-text {{
    display: flex;
    flex-direction: column;
    min-width: 0;
  }}
  .agy-brand-title {{
    font-size: 13px;
    line-height: 16px;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .agy-brand-sub {{
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
    color: var(--text-secondary);
    white-space: nowrap;
  }}

  /* Pestañas con indicador clip-path sincronizado (RECIPES.md) */
  .agy-slide-tabs {{
    display: flex;
    align-items: center;
    gap: 4px;
    background: var(--bg-canvas);
    padding: 4px;
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
  }}

  .agy-slide-tab {{
    background: transparent;
    color: var(--text-secondary);
    border: 1px solid transparent;
    padding: 6px 12px;
    border-radius: 6px;
    font-family: var(--font-sans);
    font-size: 12px;
    line-height: 16px;
    font-weight: 500;
    cursor: pointer;
    transition: transform 160ms var(--ease-out),
                color 160ms var(--ease-out),
                background-color 160ms var(--ease-out),
                border-color 160ms var(--ease-out);
    white-space: nowrap;
  }}
  .agy-slide-tab:active {{
    transform: scale(0.97);
  }}
  .agy-slide-tab.active {{
    background: var(--accent-subtle);
    color: var(--accent);
    border-color: var(--accent-border);
    font-weight: 600;
  }}

  .agy-nav-actions {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  #agy-slide-counter {{
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
    color: var(--text-secondary);
    padding: 4px 8px;
    background: var(--bg-canvas);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
  }}

  .agy-nav-btn {{
    background: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border-subtle);
    padding: 6px 12px;
    border-radius: 6px;
    font-family: var(--font-sans);
    font-size: 12px;
    line-height: 16px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: transform 160ms var(--ease-out),
                background-color 160ms var(--ease-out),
                border-color 160ms var(--ease-out);
  }}
  .agy-nav-btn:active:not(:disabled) {{
    transform: scale(0.97);
  }}
  .agy-nav-btn:disabled {{
    opacity: 0.36;
    cursor: not-allowed;
  }}

  /* ═══════════════════════════════════════════════════════════════
     CONTENEDOR PRINCIPAL DE LAS 4 DIAPOSITIVAS
     ═══════════════════════════════════════════════════════════════ */
  #agy-app-shell {{
    position: fixed;
    top: 56px;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 9500;
    background: var(--bg-canvas);
    color: var(--text-primary);
    font-family: var(--font-sans);
    display: flex;
    flex-direction: column;
    opacity: 0;
    pointer-events: none;
    transform: translateY(4px) scale(0.997);
    transition: opacity 160ms var(--ease-out),
                transform 160ms var(--ease-out);
  }}
  #agy-app-shell.visible {{
    opacity: 1;
    pointer-events: auto;
    transform: translateY(0) scale(1);
    transition: opacity 220ms var(--ease-out),
                transform 220ms var(--ease-out);
  }}
  /* Desactivar animación cuando el cambio proviene de atajo de teclado (100+/día rule) */
  #agy-app-shell.keyboard-instant {{
    transition-duration: 0ms !important;
  }}

  {estilo_puente_folium}

  /* Barra secundaria contextual (48px = 6 × 8px) */
  .agy-deck-header {{
    min-height: 52px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border-subtle);
    flex-wrap: wrap;
    gap: 12px;
  }}
  .agy-deck-title-box {{
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .agy-deck-title-box h1 {{
    margin: 0;
    font-size: 16px;
    line-height: 24px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .agy-section-tag {{
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
    padding: 2px 6px;
    border-radius: 4px;
    background: var(--accent-subtle);
    color: var(--accent);
    border: 1px solid var(--accent-border);
    font-weight: 600;
  }}
  .agy-deck-title-box p {{
    margin: 0;
    font-size: 12px;
    line-height: 16px;
    color: var(--text-secondary);
  }}

  .agy-deck-controls {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }}
  .agy-ctrl-group {{
    display: inline-flex;
    align-items: center;
    background: var(--bg-canvas);
    padding: 2px;
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
    gap: 2px;
  }}
  .agy-ctrl-btn, .agy-ctrl-select {{
    background: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border-subtle);
    padding: 6px 10px;
    border-radius: 6px;
    font-family: var(--font-sans);
    font-size: 12px;
    line-height: 16px;
    font-weight: 500;
    cursor: pointer;
    transition: transform 160ms var(--ease-out),
                border-color 160ms var(--ease-out),
                background-color 160ms var(--ease-out),
                color 160ms var(--ease-out);
  }}
  .agy-ctrl-group .agy-ctrl-btn {{
    border: none;
    background: transparent;
    color: var(--text-secondary);
    padding: 4px 10px;
  }}
  .agy-ctrl-btn:active {{
    transform: scale(0.97);
  }}
  .agy-ctrl-btn.active, .agy-ctrl-group .agy-ctrl-btn.active {{
    background: var(--accent-subtle);
    border: 1px solid var(--accent-border);
    color: var(--accent);
    font-weight: 600;
  }}

  /* Cuerpo principal: Lienzo interactivo + Inspector lateral de 424px (53 × 8px) */
  .agy-deck-body {{
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 424px;
    min-height: 0;
    overflow: hidden;
  }}

  .agy-canvas-wrap {{
    position: relative;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background-color: var(--bg-canvas);
    background-image:
      linear-gradient(rgba(255, 255, 255, 0.022) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255, 255, 255, 0.022) 1px, transparent 1px);
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

  /* Tarjetas KPI superiores (Grilla 8px/12px) */
  .agy-kpi-strip {{
    position: absolute;
    top: 12px;
    left: 12px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    pointer-events: none;
    z-index: 20;
  }}
  .agy-kpi-card {{
    background: rgba(17, 20, 26, 0.90);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 8px 12px;
    backdrop-filter: blur(8px);
  }}
  .agy-kpi-val {{
    font-family: var(--font-mono);
    font-size: 15px;
    line-height: 20px;
    font-weight: 700;
    color: var(--text-primary);
  }}
  .agy-kpi-lbl {{
    font-size: 11px;
    line-height: 16px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}

  /* Panel Flotante de Capas SIG (Exclusivo Diapositiva 1: Mapa Geográfico) */
  #agy-gis-layers-panel {{
    position: absolute;
    top: 76px;
    left: 12px;
    width: 240px;
    background: rgba(17, 20, 26, 0.92);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 12px;
    backdrop-filter: blur(8px);
    z-index: 25;
    display: none;
  }}
  #agy-gis-layers-panel.visible {{
    display: block;
  }}
  .agy-gis-panel-title {{
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin: 0 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .agy-layer-toggle {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 4px 0;
    font-size: 12px;
    line-height: 16px;
    color: var(--text-primary);
    cursor: pointer;
  }}
  .agy-layer-toggle input {{
    accent-color: var(--accent);
    cursor: pointer;
  }}
  .agy-layer-count {{
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-secondary);
  }}
  .agy-cam-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px;
    margin-top: 8px;
  }}
  .agy-cam-btn {{
    background: var(--bg-elevated);
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    padding: 4px 6px;
    font-size: 11px;
    line-height: 16px;
    cursor: pointer;
    text-align: left;
    transition: transform 160ms var(--ease-out),
                color 160ms var(--ease-out),
                border-color 160ms var(--ease-out);
  }}
  .agy-cam-btn:active {{
    transform: scale(0.97);
  }}

  /* HUD Cartográfico Inferior Derecha (Coordenadas GPS + Escala Métrica) */
  #agy-coords-hud {{
    position: absolute;
    top: 12px;
    right: 12px;
    background: rgba(17, 20, 26, 0.90);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 6px 10px;
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 12px;
    pointer-events: none;
    z-index: 20;
  }}

  /* Barra de filtro rápido por tipo de vértice */
  .agy-type-filter-bar {{
    position: absolute;
    bottom: 12px;
    left: 12px;
    background: rgba(17, 20, 26, 0.92);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 12px;
    line-height: 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    max-width: calc(100% - 24px);
    backdrop-filter: blur(8px);
    z-index: 20;
  }}
  .agy-filter-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--text-secondary);
    background: transparent;
    border: 1px solid transparent;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    line-height: 16px;
    cursor: pointer;
    transition: transform 160ms var(--ease-out),
                background-color 160ms var(--ease-out),
                border-color 160ms var(--ease-out),
                color 160ms var(--ease-out);
  }}
  .agy-filter-pill:active {{
    transform: scale(0.97);
  }}
  .agy-filter-pill.active {{
    background: var(--bg-elevated);
    border-color: var(--border-strong);
    color: var(--text-primary);
    font-weight: 600;
  }}
  .agy-leg-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
  }}
  .agy-leg-line {{
    width: 18px;
    height: 0;
    display: inline-block;
  }}

  /* Panel lateral derecho: Inspector Técnico de Nodos, Aristas y Escenarios */
  .agy-inspector {{
    background: var(--bg-panel);
    border-left: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }}
  .agy-inspector-header {{
    padding: 12px 16px;
    background: var(--bg-elevated);
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }}
  .agy-inspector-header h2 {{
    margin: 0;
    font-size: 13px;
    line-height: 20px;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .agy-inspector-content {{
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    font-size: 12px;
    line-height: 18px;
  }}

  /* Cascada escalonada (Stagger 30-50ms — STANDARDS.md) */
  @keyframes agyStaggerIn {{
    from {{
      opacity: 0;
      transform: translateY(6px);
    }}
    to {{
      opacity: 1;
      transform: translateY(0);
    }}
  }}

  .agy-prop-group {{
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    animation: agyStaggerIn 220ms var(--ease-out) both;
  }}
  .agy-prop-group:nth-child(1) {{ animation-delay: 0ms; }}
  .agy-prop-group:nth-child(2) {{ animation-delay: 35ms; }}
  .agy-prop-group:nth-child(3) {{ animation-delay: 70ms; }}
  .agy-prop-group:nth-child(4) {{ animation-delay: 105ms; }}
  .agy-prop-group:nth-child(5) {{ animation-delay: 140ms; }}

  .agy-prop-group h4 {{
    margin: 0 0 8px 0;
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .agy-prop-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    gap: 8px;
  }}
  .agy-prop-row:last-child {{
    border-bottom: none;
  }}
  .agy-prop-key {{
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
  }}
  .agy-prop-val {{
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
    font-weight: 600;
    text-align: right;
  }}

  .agy-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
    font-weight: 600;
    border: 1px solid var(--border-subtle);
    flex-shrink: 0;
  }}

  .agy-effect-box {{
    background: var(--accent-subtle);
    border: 1px solid var(--accent-border);
    border-left: 3px solid var(--accent);
    padding: 10px 12px;
    border-radius: 8px;
    color: var(--text-primary);
    font-size: 12px;
    line-height: 18px;
    margin-bottom: 12px;
    animation: agyStaggerIn 180ms var(--ease-out) both;
  }}

  .agy-code-block {{
    background: var(--bg-canvas);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 10px 12px;
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
    color: #cbd5e1;
    overflow-x: auto;
    white-space: pre;
    max-height: 220px;
  }}

  .agy-chip-list {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
  }}
  .agy-chip {{
    background: var(--bg-canvas);
    border: 1px solid var(--border-subtle);
    color: var(--text-primary);
    padding: 4px 8px;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
    cursor: pointer;
    transition: transform 160ms var(--ease-out),
                border-color 160ms var(--ease-out),
                background-color 160ms var(--ease-out),
                color 160ms var(--ease-out);
  }}
  .agy-chip:active {{
    transform: scale(0.96);
  }}

  /* Tabla técnica compacta para Comparación de Escenarios */
  .agy-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
  }}
  .agy-table th {{
    text-align: left;
    color: var(--text-secondary);
    padding: 4px 6px;
    border-bottom: 1px solid var(--border-strong);
    font-weight: 600;
  }}
  .agy-table td {{
    padding: 5px 6px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    color: var(--text-primary);
  }}

  /* Barras de progreso de la fórmula multicriterio usando transform: scaleX (compositor GPU) */
  @keyframes agyScaleBar {{
    from {{ transform: scaleX(0); }}
    to {{ transform: scaleX(1); }}
  }}
  .agy-bar-wrap {{
    margin: 6px 0;
  }}
  .agy-bar-label {{
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
    color: var(--text-secondary);
  }}
  .agy-bar-bg {{
    height: 6px;
    background: var(--bg-canvas);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 4px;
  }}
  .agy-bar-fill {{
    height: 100%;
    background: var(--accent);
    border-radius: 3px;
    transform-origin: left center;
    animation: agyScaleBar 260ms var(--ease-out) both;
  }}

  /* Elementos SVG del grafo y del mapa SIG */
  .agy-edge-path {{
    fill: none;
    cursor: pointer;
    transition: stroke-width 160ms var(--ease-out),
                stroke-opacity 160ms var(--ease-out);
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
    transition: transform 160ms var(--ease-out),
                stroke-width 160ms var(--ease-out);
  }}
  .agy-node-group:active .agy-node-circle {{
    transform: scale(0.95);
  }}

  @keyframes agyDashFlow {{
    to {{ stroke-dashoffset: -36; }}
  }}
  .agy-route-highlight {{
    animation: agyDashFlow 1.1s linear infinite;
  }}

  /* ═══════════════════════════════════════════════════════════════
     ACCESIBILIDAD: HOVER GATING & PREFERS-REDUCED-MOTION (STANDARDS.md)
     ═══════════════════════════════════════════════════════════════ */
  @media (hover: hover) and (pointer: fine) {{
    .agy-nav-btn:hover:not(:disabled) {{
      background: var(--bg-hover);
      border-color: var(--border-strong);
    }}
    .agy-slide-tab:hover:not(.active) {{
      color: var(--text-primary);
      background: rgba(255, 255, 255, 0.04);
    }}
    .agy-ctrl-btn:hover:not(.active), .agy-ctrl-select:hover, .agy-cam-btn:hover {{
      border-color: var(--border-strong);
      color: var(--text-primary);
      background: var(--bg-hover);
    }}
    .agy-chip:hover {{
      border-color: var(--accent-border);
      background: var(--accent-subtle);
      color: var(--accent);
    }}
    .agy-filter-pill:hover {{
      background: var(--bg-elevated);
      color: var(--text-primary);
    }}
    .agy-edge-path:hover {{
      stroke-width: 4.2px !important;
      stroke-opacity: 1 !important;
    }}
    .agy-node-group:hover .agy-node-circle {{
      transform: scale(1.14);
      stroke-width: 3.2px;
    }}
  }}

  @media (prefers-reduced-motion: reduce) {{
    #agy-app-shell, .agy-prop-group, .agy-effect-box, .agy-bar-fill,
    .agy-nav-btn, .agy-slide-tab, .agy-ctrl-btn, .agy-chip, .agy-node-circle {{
      animation: none !important;
      transition: opacity 150ms ease, background-color 150ms ease, border-color 150ms ease !important;
      transform: none !important;
    }}
    .agy-route-highlight {{
      animation: none !important;
    }}
  }}

  /* Adaptación Responsive para pantallas compactas o tablets */
  @media (max-width: 1024px) {{
    .agy-brand-sub {{ display: none; }}
    .agy-deck-body {{
      grid-template-columns: 1fr;
      grid-template-rows: 1fr 296px;
    }}
    .agy-inspector {{
      border-left: none;
      border-top: 1px solid var(--border-subtle);
    }}
  }}
</style>

<!-- ═══════════════════════════════════════════════════════════════════════
     BARRA SUPERIOR GLOBAL DE NAVEGACIÓN (4 DIAPOSITIVAS)
     ═══════════════════════════════════════════════════════════════════════ -->
<header id="agy-slide-nav" role="navigation" aria-label="Navegación de presentación logística">
  <div class="agy-brand">
    <div class="agy-brand-mark" aria-hidden="true">G(V,E)</div>
    <div class="agy-brand-text">
      <span class="agy-brand-title">Red Logística Multimodal · Pesca Artesanal</span>
      <span class="agy-brand-sub">Puerto Gaitán, Meta → Villavicencio / Bogotá · 25 Vértices · 41 Aristas</span>
    </div>
  </div>

  <div class="agy-slide-tabs" role="tablist" aria-label="Diapositivas">
    <button class="agy-slide-tab active" role="tab" aria-selected="true" data-slide="0" onclick="agyCambiarSlide(0, false)">
      1. 🗺️ Mapa Geográfico
    </button>
    <button class="agy-slide-tab" role="tab" aria-selected="false" data-slide="1" onclick="agyCambiarSlide(1, false)">
      2. 🛣️ Grafo Vial
    </button>
    <button class="agy-slide-tab" role="tab" aria-selected="false" data-slide="2" onclick="agyCambiarSlide(2, false)">
      3. 🌊 Grafo Fluvial
    </button>
    <button class="agy-slide-tab" role="tab" aria-selected="false" data-slide="3" onclick="agyCambiarSlide(3, false)">
      4. 🔗 Grafo Mixto Multimodal
    </button>
  </div>

  <div class="agy-nav-actions">
    <span id="agy-slide-counter" aria-live="polite">Vista 1 / 4</span>
    <button id="agy-btn-prev" class="agy-nav-btn" onclick="agyCambiarSlide(agySlideActual - 1, false)" disabled aria-label="Diapositiva anterior">
      ◀ Anterior
    </button>
    <button id="agy-btn-next" class="agy-nav-btn" onclick="agyCambiarSlide(agySlideActual + 1, false)" aria-label="Siguiente diapositiva">
      Siguiente ▶
    </button>
  </div>
</header>

<!-- ═══════════════════════════════════════════════════════════════════════
     SHELL INTERACTIVO DE LAS 4 DIAPOSITIVAS
     ═══════════════════════════════════════════════════════════════════════ -->
<main id="agy-app-shell" class="{clase_slide0}">
  <section class="agy-deck-header" aria-label="Controles de visualización">
    <div class="agy-deck-title-box">
      <h1>
        <span id="agy-slide-tag" class="agy-section-tag">SIG VECTORIAL</span>
        <span id="agy-slide-title">Diapositiva 1: Mapa Geográfico de la Cadena Logística</span>
      </h1>
      <p id="agy-slide-subtitle">Geometrías reales offline: red hidrográfica rios.shp (12,413 vértices), rutas viales OSRM, 25 nodos del modelo y 49 puntos de campo.</p>
    </div>

    <div class="agy-deck-controls">
      <!-- Conmutador de Disposición (activo en Diapositivas 2, 3 y 4) -->
      <div id="agy-layout-switcher" class="agy-ctrl-group" style="display:none;">
        <button id="agy-layout-hier" class="agy-ctrl-btn active" onclick="agyCambiarLayout('hier')">
          🧬 Jerárquica (Etapas)
        </button>
        <button id="agy-layout-geo" class="agy-ctrl-btn" onclick="agyCambiarLayout('geo')">
          📍 Geográfica (Lupa Focal)
        </button>
      </div>

      <!-- Conmutador de Destino de Itinerarios -->
      <div class="agy-ctrl-group" title="Seleccionar mercado objetivo para itinerarios y escenarios">
        <button id="agy-dest-bog" class="agy-ctrl-btn active" onclick="agyCambiarDestino('BOG')">
          Destino: Bogotá
        </button>
        <button id="agy-dest-pg" class="agy-ctrl-btn" onclick="agyCambiarDestino('PG')">
          Destino: Pto. Gaitán
        </button>
      </div>

      <!-- Selector de Itinerario -->
      <select id="agy-itin-select" class="agy-ctrl-select" aria-label="Resaltar itinerario de pesca" onchange="agySeleccionarItinerario(this.value)">
        <option value="">✨ Resaltar Itinerario Óptimo...</option>
      </select>

      <!-- Buscador / Salto Rápido a Vértice o Arista -->
      <select id="agy-quick-select" class="agy-ctrl-select" aria-label="Ir a vértice o arista" onchange="agySeleccionarElementoRapido(this.value)">
        <option value="">🔍 Inspeccionar Vértice / Arista...</option>
      </select>

      <button class="agy-ctrl-btn" onclick="agyMostrarComparadorEscenarios()" title="Ver tabla comparativa de los 5 escenarios de refrigeración">
        📊 Escenarios FLP
      </button>

      <button class="agy-ctrl-btn" onclick="agyResetVista()" title="Restaurar zoom y encuadre (Tecla R)">
        ⟲ Centrar
      </button>
    </div>
  </section>

  <div class="agy-deck-body">
    <!-- Lienzo Vectorial Interactivo (SIG en Slide 1 / Grafos en Slides 2, 3, 4) -->
    <div class="agy-canvas-wrap" id="agy-canvas-wrap">
      <!-- Tarjetas KPI en vivo -->
      <div class="agy-kpi-strip" id="agy-kpi-strip"></div>

      <!-- HUD de Coordenadas y Escala -->
      <div id="agy-coords-hud">
        <span id="agy-hud-pos">LAT 4.3130°N · LON 72.0830°W</span>
        <span id="agy-hud-zoom">ZOOM 1.0×</span>
      </div>

      <!-- Panel de Capas y Encuadres SIG (Diapositiva 1) -->
      <aside id="agy-gis-layers-panel" class="visible" aria-label="Capas del Mapa Geográfico">
        <div class="agy-gis-panel-title">
          <span>Capas Geográficas SIG</span>
          <span style="color:var(--accent)">OFFLINE</span>
        </div>
        <label class="agy-layer-toggle">
          <span><input type="checkbox" id="chk-layer-rivers" checked onchange="agyToggleCapaGIS()"> Red Fluvial (rios.shp)</span>
          <span class="agy-layer-count">12,413 V</span>
        </label>
        <label class="agy-layer-toggle">
          <span><input type="checkbox" id="chk-layer-roads" checked onchange="agyToggleCapaGIS()"> Red Vial Real (OSRM)</span>
          <span class="agy-layer-count">31 E</span>
        </label>
        <label class="agy-layer-toggle">
          <span><input type="checkbox" id="chk-layer-navrivers" checked onchange="agyToggleCapaGIS()"> Rutas Fluviales Navegables</span>
          <span class="agy-layer-count">10 E</span>
        </label>
        <label class="agy-layer-toggle">
          <span><input type="checkbox" id="chk-layer-nodes" checked onchange="agyToggleCapaGIS()"> Nodos del Modelo</span>
          <span class="agy-layer-count">25 V</span>
        </label>
        <label class="agy-layer-toggle">
          <span><input type="checkbox" id="chk-layer-field" checked onchange="agyToggleCapaGIS()"> Puntos de Campo (GPS)</span>
          <span class="agy-layer-count">49 pts</span>
        </label>
        <label class="agy-layer-toggle">
          <span><input type="checkbox" id="chk-layer-grid" checked onchange="agyToggleCapaGIS()"> Retícula WGS84</span>
          <span class="agy-layer-count">Grados</span>
        </label>

        <div class="agy-gis-panel-title" style="margin-top:10px;">
          <span>Encuadres Rápidos</span>
        </div>
        <div class="agy-cam-grid">
          <button class="agy-cam-btn" onclick="agyEnfocarRegionGIS('all')">🌎 Meta – Bogotá</button>
          <button class="agy-cam-btn" onclick="agyEnfocarRegionGIS('cuenca')">🌊 Cuenca Pto. Gaitán</button>
          <button class="agy-cam-btn" onclick="agyEnfocarRegionGIS('urbano')">🏙️ Casco Urbano</button>
          <button class="agy-cam-btn" onclick="agyEnfocarRegionGIS('rutameta')">🛣️ Corredor Ruta Meta</button>
        </div>
      </aside>

      <!-- Lienzo SVG Principal -->
      <svg id="agy-graph-svg" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Visualización interactiva de red logística">
        <defs>
          <marker id="arrow-terrestre" viewBox="0 0 10 10" refX="21" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
            <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#22c55e"/>
          </marker>
          <marker id="arrow-regular" viewBox="0 0 10 10" refX="21" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
            <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#f97316"/>
          </marker>
          <marker id="arrow-mala" viewBox="0 0 10 10" refX="21" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
            <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#ef4444"/>
          </marker>
          <marker id="arrow-acceso" viewBox="0 0 10 10" refX="21" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
            <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#9ca3af"/>
          </marker>
          <marker id="arrow-fluvial" viewBox="0 0 10 10" refX="21" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
            <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#38bdf8"/>
          </marker>
          <marker id="arrow-active" viewBox="0 0 10 10" refX="21" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#facc15"/>
          </marker>
        </defs>
        <g id="agy-viewport">
          <g id="agy-layer-guides"></g>
          <g id="agy-layer-rivers"></g>
          <g id="agy-layer-fieldpoints"></g>
          <g id="agy-layer-edges"></g>
          <g id="agy-layer-nodes"></g>
        </g>
      </svg>

      <!-- Barra Inferior de Filtrado por Tipo de Vértice y Leyenda Semántica -->
      <div class="agy-type-filter-bar" role="toolbar" aria-label="Filtrar por tipo de nodo">
        <button class="agy-filter-pill active" data-tipo="ALL" onclick="agyFiltrarTipoNodo('ALL')">Todos (25)</button>
        <button class="agy-filter-pill" data-tipo="P" onclick="agyFiltrarTipoNodo('P')"><span class="agy-leg-dot" style="background:var(--col-pesca)"></span>Pesca P (7)</button>
        <button class="agy-filter-pill" data-tipo="M" onclick="agyFiltrarTipoNodo('M')"><span class="agy-leg-dot" style="background:var(--col-muelle)"></span>Muelle M (3)</button>
        <button class="agy-filter-pill" data-tipo="A" onclick="agyFiltrarTipoNodo('A')"><span class="agy-leg-dot" style="background:var(--col-acopio)"></span>Acopio A (3)</button>
        <button class="agy-filter-pill" data-tipo="F" onclick="agyFiltrarTipoNodo('F')"><span class="agy-leg-dot" style="background:var(--col-frio)"></span>Frío F (3)</button>
        <button class="agy-filter-pill" data-tipo="D" onclick="agyFiltrarTipoNodo('D')"><span class="agy-leg-dot" style="background:var(--col-deterioro)"></span>Deterioro D (2)</button>
        <button class="agy-filter-pill" data-tipo="H" onclick="agyFiltrarTipoNodo('H')"><span class="agy-leg-dot" style="background:var(--col-habitat)"></span>Hábitat H (2)</button>
        <button class="agy-filter-pill" data-tipo="V" onclick="agyFiltrarTipoNodo('V')"><span class="agy-leg-dot" style="background:var(--col-venta)"></span>Venta V (5)</button>
        <span style="color:var(--border-strong)">|</span>
        <span class="agy-filter-pill" style="cursor:default"><span class="agy-leg-line" style="border-top:2.5px dashed #38bdf8"></span>Fluvial</span>
        <span class="agy-filter-pill" style="cursor:default"><span class="agy-leg-line" style="border-top:2.5px solid #9ca3af"></span>Acceso Local</span>
        <span class="agy-filter-pill" style="cursor:default"><span class="agy-leg-line" style="border-top:2.5px solid #22c55e"></span>Terrestre</span>
      </div>
    </div>

    <!-- Inspector Técnico de Datos (Derecha) -->
    <aside class="agy-inspector" aria-label="Inspector de propiedades del elemento seleccionado">
      <div class="agy-inspector-header">
        <h2 id="agy-insp-title">🔎 Inspector de Estructura de Datos</h2>
        <span id="agy-insp-badge" class="agy-badge" style="background:var(--bg-canvas);color:var(--accent);">Grafo Activo</span>
      </div>
      <div class="agy-inspector-content" id="agy-insp-body"></div>
    </aside>
  </div>
</main>

<script>
const AGY_DATA = {json_payload};
const AGY_STANDALONE = {"true" if modo_autonomo else "false"};

const AGY_COLORES_NODO = {{
  P: "#38bdf8",
  M: "#a855f7",
  A: "#f59e0b",
  F: "#06b6d4",
  D: "#ef4444",
  H: "#10b981",
  V: "#ec4899"
}};

const AGY_ETAPAS_JERARQUICAS = [
  {{ col: 0, titulo: "1. CAPTURA (P)" }},
  {{ col: 1, titulo: "2. TRANSBORDO (M)" }},
  {{ col: 2, titulo: "3. ACOPIO (A)" }},
  {{ col: 3, titulo: "4. CONTROL (D / H)" }},
  {{ col: 4, titulo: "5. REFRIGERACIÓN (F)" }},
  {{ col: 5, titulo: "6. DESTINO (V)" }}
];

// Estado separado de la geometría y del DOM (Section 6)
let agySlideActual = 0;
let agyLayoutModo = "hier";      // 'hier' | 'geo'
let agyDestinoModo = "BOG";      // 'BOG' | 'PG'
let agyFiltroTipoNodo = "ALL";   // 'ALL' | 'P' | 'M' | 'A' | 'F' | 'D' | 'H' | 'V'
let agySeleccion = null;         // {{ tipo: 'nodo'|'arista'|'campo', id: ... }}
let agyHoverNodeId = null;
let agyItinerarioActivo = "";
let agyPosiciones = {{}};
let agyTargets = {{}};
let agySpringAnimId = null;
let agyTransform = {{ x: 0, y: 0, k: 1 }};
let agyDraggingNode = null;
let agyDragMoved = false;
let agyPanning = false;
let agyPanStart = {{ x: 0, y: 0 }};

// Referencias DOM cacheadas para actualización geométrica a 60fps sin reconstruir el árbol SVG
let agyDomNodes = {{}};
let agyDomEdges = {{}};

function agyPrefersReducedMotion() {{
  return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}}

function agyListaItinerariosActiva() {{
  return agyDestinoModo === "PG" ? (AGY_DATA.itinerarios_locales || []) : (AGY_DATA.itinerarios || []);
}}

function agyPoblarSelectorItinerarios() {{
  const selIt = document.getElementById("agy-itin-select");
  const valorPrevio = selIt.value;
  selIt.innerHTML = `<option value="">✨ Resaltar Itinerario (${{agyDestinoModo === "PG" ? "Pto. Gaitán" : "Bogotá"}})...</option>`;
  agyListaItinerariosActiva().forEach(it => {{
    const opt = document.createElement("option");
    opt.value = it.origen_id;
    opt.textContent = `🐟 ${{it.origen_id}} → ${{it.destino_id}}: ${{it.ruta.join(" → ")}} (${{it.distancia_total_km}} km)`;
    selIt.appendChild(opt);
  }});
  if (valorPrevio) selIt.value = valorPrevio;
}}

function agyCambiarDestino(modo) {{
  agyDestinoModo = modo;
  document.getElementById("agy-dest-bog").classList.toggle("active", modo === "BOG");
  document.getElementById("agy-dest-pg").classList.toggle("active", modo === "PG");
  agyPoblarSelectorItinerarios();
  if (agyItinerarioActivo) {{
    agySeleccionarItinerario(agyItinerarioActivo);
  }} else {{
    agyRenderizarEscenaCompleta();
  }}
}}

// Filtrar nodos y aristas según la diapositiva activa y filtro de tipo
function agySubgrafoActual() {{
  let aristas = [];
  if (agySlideActual === 1) {{
    // Diapositiva 2: Exclusivamente Grafo Vial (terrestre + acceso_local)
    aristas = AGY_DATA.aristas.filter(a => a.tipo_via === "terrestre" || a.tipo_via === "acceso_local");
  }} else if (agySlideActual === 2) {{
    // Diapositiva 3: Exclusivamente Grafo Fluvial (fluvial)
    aristas = AGY_DATA.aristas.filter(a => a.tipo_via === "fluvial");
  }} else {{
    // Diapositivas 1 (Mapa Geográfico SIG) y 4 (Grafo Mixto Multimodal)
    aristas = AGY_DATA.aristas.slice();
  }}

  const idsActivos = new Set();
  aristas.forEach(a => {{
    idsActivos.add(a.origen);
    idsActivos.add(a.destino);
  }});

  let nodos = AGY_DATA.nodos.filter(n => (agySlideActual === 0 || agySlideActual === 3) || idsActivos.has(n.id));
  if (agyFiltroTipoNodo !== "ALL") {{
    const nodosTipo = new Set(nodos.filter(n => n.tipo === agyFiltroTipoNodo).map(n => n.id));
    aristas = aristas.filter(a => nodosTipo.has(a.origen) || nodosTipo.has(a.destino));
    const idsVecinos = new Set(nodosTipo);
    aristas.forEach(a => {{
      idsVecinos.add(a.origen);
      idsVecinos.add(a.destino);
    }});
    nodos = nodos.filter(n => idsVecinos.has(n.id));
  }}
  return {{ nodos, aristas }};
}}

// Proyección Cartográfica Real Equirectangular para Diapositiva 1 (Mapa Geográfico SIG)
function agyProyectarSIG(lon, lat, W, H) {{
  const minLon = -74.22, maxLon = -71.00;
  const minLat = 3.22,   maxLat = 4.92;
  const padX = 56, padY = 48;
  const availW = Math.max(200, W - 2 * padX);
  const availH = Math.max(200, H - 2 * padY);
  const scaleX = availW / (maxLon - minLon);
  const scaleY = availH / (maxLat - minLat);
  const scale = Math.min(scaleX, scaleY);
  const mapW = (maxLon - minLon) * scale;
  const mapH = (maxLat - minLat) * scale;
  const offX = (W - mapW) / 2;
  const offY = (H - mapH) / 2;
  return {{
    x: offX + (lon - minLon) * scale,
    y: offY + (maxLat - lat) * scale,
    scale: scale
  }};
}}

function agyInvertirSIG(x, y, W, H) {{
  const minLon = -74.22, maxLon = -71.00;
  const minLat = 3.22,   maxLat = 4.92;
  const padX = 56, padY = 48;
  const availW = Math.max(200, W - 2 * padX);
  const availH = Math.max(200, H - 2 * padY);
  const scale = Math.min(availW / (maxLon - minLon), availH / (maxLat - minLat));
  const mapW = (maxLon - minLon) * scale;
  const mapH = (maxLat - minLat) * scale;
  const offX = (W - mapW) / 2;
  const offY = (H - mapH) / 2;
  const lon = minLon + (x - offX) / scale;
  const lat = maxLat - (y - offY) / scale;
  return {{ lon, lat }};
}}

// Proyección Geográfica con Lupa Focal + Relajación Elíptica para Diapositivas 2, 3 y 4
function agyInterpolarTramo(val, dom, rng) {{
  if (val <= dom[0]) return rng[0];
  if (val >= dom[dom.length - 1]) return rng[rng.length - 1];
  for (let i = 0; i < dom.length - 1; i++) {{
    if (val >= dom[i] && val <= dom[i + 1]) {{
      const t = (val - dom[i]) / ((dom[i + 1] - dom[i]) || 1e-9);
      return rng[i] + t * (rng[i + 1] - rng[i]);
    }}
  }}
  return rng[0];
}}

function agyProyectarGeoLupa(lon, lat, W, H, padX, padY) {{
  const domLon = [-74.10, -72.80, -72.13, -72.08, -72.03, -71.48];
  const rngLon = [0.04,   0.18,   0.27,   0.56,   0.85,   0.95];
  const nx = agyInterpolarTramo(lon, domLon, rngLon);

  const domLat = [4.08, 4.16, 4.25, 4.31, 4.36, 4.70];
  const rngLat = [0.92, 0.78, 0.62, 0.38, 0.18, 0.06];
  const ny = agyInterpolarTramo(lat, domLat, rngLat);

  return {{
    x: padX + nx * (W - 2 * padX),
    y: padY + ny * (H - 2 * padY)
  }};
}}

function agyNombreCompacto(nombre) {{
  if (!nombre) return "";
  return nombre
    .replace("ASOPESGA - ", "")
    .replace("Asovida - ", "")
    .replace("Centro Frío ", "CF ")
    .replace("Acopio ", "Ac. ")
    .replace("Muelle ", "M. ")
    .replace("Zona Deterioro Térmico ", "Det. ")
    .replace("Zona Deterioro ", "Det. ")
    .replace("Hábitat Protegido ", "Háb. ");
}}

function agyCalcularPosicionesObjetivo(inmediato) {{
  const svg = document.getElementById("agy-graph-svg");
  const W = svg.clientWidth || 960;
  const H = svg.clientHeight || 640;
  const padX = 82, padYTop = 116, padYBot = 68;
  const usableH = Math.max(220, H - padYTop - padYBot);

  if (agySlideActual === 0) {{
    // Diapositiva 1: Coordenadas cartográficas reales WGS84 + micro-desplazamiento para pines coincidentes
    AGY_DATA.nodos.forEach((n, idx) => {{
      const p = agyProyectarSIG(n.longitud, n.latitud, W, H);
      agyTargets[n.id] = {{ x: p.x, y: p.y }};
    }});
    // Evitar que nodos con idéntico GPS (P06/M03/F02/V_PG1/V_PG2, P03/A01, P05/A03) se tapen en el mapa SIG
    const idsGIS = AGY_DATA.nodos.map(n => n.id);
    const minDistGIS = 5.2;
    for (let iter = 0; iter < 45; iter++) {{
      for (let i = 0; i < idsGIS.length; i++) {{
        for (let j = i + 1; j < idsGIS.length; j++) {{
          const a = agyTargets[idsGIS[i]];
          const b = agyTargets[idsGIS[j]];
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) {{
            const ang = (i * 1.9 + j * 0.7);
            dx = Math.cos(ang) * 0.5;
            dy = Math.sin(ang) * 0.5;
          }}
          const d = Math.hypot(dx, dy);
          if (d < minDistGIS) {{
            const push = (minDistGIS - d) * 0.45;
            const ux = dx / (d || 1);
            const uy = dy / (d || 1);
            a.x -= ux * push;
            a.y -= uy * push;
            b.x += ux * push;
            b.y += uy * push;
          }}
        }}
      }}
    }}
  }} else if (agyLayoutModo === "geo") {{
    const ordenTipo = {{ P: 0, M: 1, A: 2, D: 3, H: 4, F: 5, V: 6 }};
    const basePos = {{}};
    AGY_DATA.nodos.forEach((n, idx) => {{
      const p = agyProyectarGeoLupa(n.longitud, n.latitud, W, H, padX, padYTop);
      // Ajustar Y dentro de la banda libre debajo de los KPIs (padYTop .. H - padYBot)
      const ny = (p.y - padYTop) / Math.max(1, (H - 2 * padYTop));
      const yMapped = padYTop + Math.max(0, Math.min(1, ny)) * usableH;
      const rol = ordenTipo[n.tipo] !== undefined ? ordenTipo[n.tipo] : 3;
      const ang = (idx * 2.39996) + rol * 0.55;
      basePos[n.id] = {{
        x: p.x + (rol - 2.5) * 16 + Math.cos(ang) * 12,
        y: yMapped + Math.sin(ang) * 14
      }};
      agyTargets[n.id] = {{ x: basePos[n.id].x, y: basePos[n.id].y }};
    }});

    const minSepX = Math.max(118, Math.min(148, W * 0.128));
    const minSepY = Math.max(64, Math.min(84, usableH * 0.125));
    const ids = AGY_DATA.nodos.map(n => n.id);

    for (let iter = 0; iter < 180; iter++) {{
      for (let i = 0; i < ids.length; i++) {{
        for (let j = i + 1; j < ids.length; j++) {{
          const a = agyTargets[ids[i]];
          const b = agyTargets[ids[j]];
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) {{
            dx = (i % 2 === 0 ? 1 : -1) * 4;
            dy = (j % 2 === 0 ? 1 : -1) * 4;
          }}
          const nx = dx / minSepX;
          const ny = dy / minSepY;
          const dEllipse = Math.hypot(nx, ny);
          if (dEllipse < 1.0) {{
            const push = (1.0 - dEllipse) * 0.54;
            const ux = nx / (dEllipse || 1);
            const uy = ny / (dEllipse || 1);
            a.x -= ux * minSepX * push;
            a.y -= uy * minSepY * push;
            b.x += ux * minSepX * push;
            b.y += uy * minSepY * push;
          }}
        }}
      }}
      const kAnchor = 0.028;
      ids.forEach(id => {{
        const t = agyTargets[id];
        const b = basePos[id];
        t.x += (b.x - t.x) * kAnchor;
        t.y += (b.y - t.y) * kAnchor;
        t.x = Math.max(padX, Math.min(W - padX, t.x));
        t.y = Math.max(padYTop, Math.min(H - padYBot, t.y));
      }});
    }}
  }} else {{
    // Disposición Jerárquica por etapas logísticas (P -> M -> A -> D/H -> F -> V)
    const columnas = {{ P: 0, M: 1, A: 2, D: 3, H: 3, F: 4, V: 5 }};
    const grupos = {{ 0: [], 1: [], 2: [], 3: [], 4: [], 5: [] }};
    AGY_DATA.nodos.forEach(n => {{
      const c = columnas[n.tipo] !== undefined ? columnas[n.tipo] : 3;
      grupos[c].push(n);
    }});
    Object.keys(grupos).forEach(c => {{
      const lista = grupos[c];
      const x = padX + (Number(c) / 5) * (W - 2 * padX);
      lista.forEach((n, idx) => {{
        const y = padYTop + ((idx + 0.5) / Math.max(lista.length, 1)) * usableH;
        agyTargets[n.id] = {{ x, y }};
      }});
    }});
  }}

  if (inmediato || agyPrefersReducedMotion() || Object.keys(agyPosiciones).length === 0) {{
    AGY_DATA.nodos.forEach(n => {{
      agyPosiciones[n.id] = {{ x: agyTargets[n.id].x, y: agyTargets[n.id].y }};
    }});
  }} else {{
    agyIniciarAnimacionSpring();
  }}
}}

// Geometría de curva cuadrática para una arista en modo grafo
function agyRutaCurvaArista(p1, p2) {{
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const dist = Math.hypot(dx, dy) || 1;
  const offset = Math.min(24, dist * 0.11);
  const mx = (p1.x + p2.x) / 2 - (dy / dist) * offset;
  const my = (p1.y + p2.y) / 2 + (dx / dist) * offset;
  return {{
    d: `M ${{p1.x.toFixed(1)}} ${{p1.y.toFixed(1)}} Q ${{mx.toFixed(1)}} ${{my.toFixed(1)}} ${{p2.x.toFixed(1)}} ${{p2.y.toFixed(1)}}`,
    lx: 0.25 * p1.x + 0.5 * mx + 0.25 * p2.x,
    ly: 0.25 * p1.y + 0.5 * my + 0.25 * p2.y - 4
  }};
}}

// Actualiza ÚNICAMENTE las coordenadas geométricas en el DOM existente (60fps sin destruir nodos)
function agyActualizarGeometriaDOM() {{
  Object.keys(agyDomNodes).forEach(id => {{
    const pos = agyPosiciones[id];
    const g = agyDomNodes[id];
    if (pos && g) {{
      g.setAttribute("transform", `translate(${{pos.x.toFixed(1)}}, ${{pos.y.toFixed(1)}})`);
    }}
  }});

  if (agySlideActual !== 0) {{
    Object.keys(agyDomEdges).forEach(edgeId => {{
      const ref = agyDomEdges[edgeId];
      const p1 = agyPosiciones[ref.arista.origen];
      const p2 = agyPosiciones[ref.arista.destino];
      if (!p1 || !p2) return;
      const geom = agyRutaCurvaArista(p1, p2);
      ref.pathVis.setAttribute("d", geom.d);
      ref.pathHit.setAttribute("d", geom.d);
      if (ref.txt) {{
        ref.txt.setAttribute("x", geom.lx.toFixed(1));
        ref.txt.setAttribute("y", geom.ly.toFixed(1));
      }}
    }});
  }}
}}

function agyIniciarAnimacionSpring() {{
  if (agySpringAnimId) cancelAnimationFrame(agySpringAnimId);
  const factor = 0.25;

  function step() {{
    let maxDelta = 0;
    AGY_DATA.nodos.forEach(n => {{
      const cur = agyPosiciones[n.id];
      const tgt = agyTargets[n.id];
      if (!cur || !tgt) return;
      const dx = tgt.x - cur.x;
      const dy = tgt.y - cur.y;
      maxDelta = Math.max(maxDelta, Math.abs(dx), Math.abs(dy));
      cur.x += dx * factor;
      cur.y += dy * factor;
    }});
    agyActualizarGeometriaDOM();
    if (maxDelta > 0.35) {{
      agySpringAnimId = requestAnimationFrame(step);
    }} else {{
      AGY_DATA.nodos.forEach(n => {{
        if (agyTargets[n.id]) {{
          agyPosiciones[n.id].x = agyTargets[n.id].x;
          agyPosiciones[n.id].y = agyTargets[n.id].y;
        }}
      }});
      agyActualizarGeometriaDOM();
      agySpringAnimId = null;
    }}
  }}
  agySpringAnimId = requestAnimationFrame(step);
}}

function agyActualizarTransform() {{
  const vp = document.getElementById("agy-viewport");
  if (vp) {{
    vp.setAttribute("transform", `translate(${{agyTransform.x.toFixed(1)}}, ${{agyTransform.y.toFixed(1)}}) scale(${{agyTransform.k.toFixed(3)}})`);
  }}
  const hudZoom = document.getElementById("agy-hud-zoom");
  if (hudZoom) {{
    hudZoom.textContent = `ZOOM ${{agyTransform.k.toFixed(1)}}×`;
  }}
}}

// Encuadres predefinidos del Mapa Geográfico SIG (Diapositiva 1)
function agyEnfocarRegionGIS(region) {{
  const svg = document.getElementById("agy-graph-svg");
  const W = svg.clientWidth || 960;
  const H = svg.clientHeight || 640;

  if (region === "all") {{
    agyTransform = {{ x: 0, y: 0, k: 1 }};
  }} else if (region === "cuenca") {{
    const p = agyProyectarSIG(-72.08, 4.30, W, H);
    const k = 4.2;
    agyTransform = {{ x: W / 2 - p.x * k, y: H / 2 - p.y * k, k }};
  }} else if (region === "urbano") {{
    const p = agyProyectarSIG(-72.083, 4.313, W, H);
    const k = 10.5;
    agyTransform = {{ x: W / 2 - p.x * k, y: H / 2 - p.y * k, k }};
  }} else if (region === "rutameta") {{
    const p = agyProyectarSIG(-72.08, 4.19, W, H);
    const k = 5.5;
    agyTransform = {{ x: W / 2 - p.x * k, y: H / 2 - p.y * k, k }};
  }}
  agyActualizarTransform();
}}

function agyResetVista() {{
  if (agySlideActual === 0) {{
    agyEnfocarRegionGIS("cuenca");
  }} else {{
    agyTransform = {{ x: 0, y: 0, k: 1 }};
    agyActualizarTransform();
    agyCalcularPosicionesObjetivo(false);
  }}
}}

function agyCambiarLayout(modo) {{
  agyLayoutModo = modo;
  document.getElementById("agy-layout-geo").classList.toggle("active", modo === "geo");
  document.getElementById("agy-layout-hier").classList.toggle("active", modo === "hier");
  agyTransform = {{ x: 0, y: 0, k: 1 }};
  agyActualizarTransform();
  agyRenderizarEscenaCompleta(false);
}}

function agyFiltrarTipoNodo(tipo) {{
  agyFiltroTipoNodo = tipo;
  document.querySelectorAll(".agy-filter-pill[data-tipo]").forEach(btn => {{
    btn.classList.toggle("active", btn.dataset.tipo === tipo);
  }});
  agyRenderizarEscenaCompleta(true);
}}

function agyToggleCapaGIS() {{
  if (agySlideActual === 0) {{
    agyRenderizarEscenaCompleta(true);
  }}
}}

function agyCambiarSlide(nuevoSlide, viaKeyboard) {{
  if (nuevoSlide < 0 || nuevoSlide > 3) return;
  agySlideActual = nuevoSlide;

  document.getElementById("agy-btn-prev").disabled = (agySlideActual === 0);
  document.getElementById("agy-btn-next").disabled = (agySlideActual === 3);
  document.getElementById("agy-slide-counter").textContent = `Vista ${{agySlideActual + 1}} / 4`;

  document.querySelectorAll(".agy-slide-tab").forEach(btn => {{
    const activo = Number(btn.dataset.slide) === agySlideActual;
    btn.classList.toggle("active", activo);
    btn.setAttribute("aria-selected", activo ? "true" : "false");
  }});

  const shell = document.getElementById("agy-app-shell");
  if (viaKeyboard) {{
    shell.classList.add("keyboard-instant");
  }} else {{
    shell.classList.remove("keyboard-instant");
  }}

  if (!AGY_STANDALONE && agySlideActual === 0) {{
    shell.classList.remove("visible");
    return;
  }}
  shell.classList.add("visible");

  const metaSlides = {{
    0: {{
      tag: "SIG VECTORIAL OFFLINE",
      titulo: "Diapositiva 1: Mapa Geográfico de la Cadena Logística",
      sub: "Geometrías reales offline: red hidrográfica rios.shp (12,413 vértices), trazados viales OSRM, 25 nodos del modelo y 49 puntos de campo."
    }},
    1: {{
      tag: "SUBGRAFO VIAL",
      titulo: "Diapositiva 2: Grafo Vial (Red Terrestre y Accesos Locales)",
      sub: "Exclusivamente las 31 conexiones terrestres (19 carreteras nacionales/secundarias y 12 accesos locales/trochas) y sus 19 vértices."
    }},
    2: {{
      tag: "SUBGRAFO FLUVIAL",
      titulo: "Diapositiva 3: Grafo Fluvial (Rutas Navegables vs Hidrografía)",
      sub: "Diferencia las 10 aristas de transporte fluvial navegable frente a la geometría hidrográfica de 12,413 vértices de rios.shp."
    }},
    3: {{
      tag: "RED MULTIMODAL",
      titulo: "Diapositiva 4: Grafo Mixto Multimodal (Pesca → Transbordo → Acopio → Frío → Venta)",
      sub: "Integra los 25 vértices, las 41 aristas dirigidas, nodos de transferencia modal y evaluación de escenarios de refrigeración."
    }}
  }};

  document.getElementById("agy-slide-tag").textContent = metaSlides[agySlideActual].tag;
  document.getElementById("agy-slide-title").textContent = metaSlides[agySlideActual].titulo;
  document.getElementById("agy-slide-subtitle").textContent = metaSlides[agySlideActual].sub;

  // Mostrar/ocultar controles según corresponda a Mapa SIG (Slide 0) o Grafos (Slides 1..3)
  document.getElementById("agy-gis-layers-panel").classList.toggle("visible", agySlideActual === 0);
  document.getElementById("agy-layout-switcher").style.display = (agySlideActual === 0) ? "none" : "inline-flex";

  if (agySlideActual === 0) {{
    agyEnfocarRegionGIS("cuenca");
  }} else {{
    agyTransform = {{ x: 0, y: 0, k: 1 }};
    agyActualizarTransform();
  }}

  agyRenderizarEscenaCompleta(true);
  agyActualizarSelectorRapido();
  if (!agySeleccion) {{
    agyMostrarResumenSubgrafo();
  }}
}}

function agyActualizarSelectorRapido() {{
  const {{ nodos, aristas }} = agySubgrafoActual();
  const sel = document.getElementById("agy-quick-select");
  sel.innerHTML = '<option value="">🔍 Inspeccionar Vértice / Arista...</option>';

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
  const idxSep = val.indexOf(":");
  const tipo = val.slice(0, idxSep);
  const id = val.slice(idxSep + 1);
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
  agyRenderizarEscenaCompleta(true);
  if (origenId) {{
    const it = agyListaItinerariosActiva().find(x => x.origen_id === origenId);
    if (it) agyInspeccionarItinerario(it);
  }} else {{
    agyMostrarResumenSubgrafo();
  }}
}}

// Construye la escena SVG para la diapositiva actual
function agyRenderizarEscenaCompleta(inmediato = true) {{
  agyCalcularPosicionesObjetivo(inmediato);

  const {{ nodos, aristas }} = agySubgrafoActual();
  const layerGuides = document.getElementById("agy-layer-guides");
  const layerRivers = document.getElementById("agy-layer-rivers");
  const layerField = document.getElementById("agy-layer-fieldpoints");
  const layerEdges = document.getElementById("agy-layer-edges");
  const layerNodes = document.getElementById("agy-layer-nodes");

  layerGuides.innerHTML = "";
  layerRivers.innerHTML = "";
  layerField.innerHTML = "";
  layerEdges.innerHTML = "";
  layerNodes.innerHTML = "";
  agyDomNodes = {{}};
  agyDomEdges = {{}};

  const svg = document.getElementById("agy-graph-svg");
  const W = svg.clientWidth || 960;
  const H = svg.clientHeight || 640;
  const padX = 76, padY = 80;

  // 1. Actualizar KPIs superiores
  const kpiStrip = document.getElementById("agy-kpi-strip");
  const kmTotal = aristas.reduce((s, a) => s + a.distancia_km, 0);
  const fluvCount = aristas.filter(a => a.tipo_via === "fluvial").length;
  const vialCount = aristas.filter(a => a.tipo_via !== "fluvial").length;
  const multimodalCount = nodos.filter(n => n.es_multimodal).length;

  kpiStrip.innerHTML = `
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{nodos.length}}</div>
      <div class="agy-kpi-lbl">Vértices |V|</div>
    </div>
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{aristas.length}}</div>
      <div class="agy-kpi-lbl">Aristas |E|</div>
    </div>
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{vialCount}} / ${{fluvCount}}</div>
      <div class="agy-kpi-lbl">Vial / Fluvial</div>
    </div>
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{kmTotal.toFixed(1)}} km</div>
      <div class="agy-kpi-lbl">Longitud Red</div>
    </div>
    ${{agySlideActual === 0 ? `
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{(AGY_DATA.puntos_campo || []).length}}</div>
      <div class="agy-kpi-lbl">Puntos Campo</div>
    </div>` : ""}}
    ${{agySlideActual === 2 ? `
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{AGY_DATA.stats_rio.vertices.toLocaleString()}}</div>
      <div class="agy-kpi-lbl">Vértices rios.shp</div>
    </div>` : ""}}
    ${{agySlideActual === 3 ? `
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{multimodalCount}}</div>
      <div class="agy-kpi-lbl">Nodos Transbordo</div>
    </div>` : ""}}
  `;

  // 2. Capas de Fondo: Retícula Cartográfica WGS84 (Slide 0) o Guías de Layout (Slides 1..3)
  if (agySlideActual === 0) {{
    const showGrid = document.getElementById("chk-layer-grid")?.checked ?? true;
    const showRivers = document.getElementById("chk-layer-rivers")?.checked ?? true;
    const showField = document.getElementById("chk-layer-field")?.checked ?? true;

    if (showGrid) {{
      for (let lon = -74.0; lon <= -71.0; lon += 0.5) {{
        const pTop = agyProyectarSIG(lon, 4.90, W, H);
        const pBot = agyProyectarSIG(lon, 3.25, W, H);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", pTop.x);
        line.setAttribute("y1", pTop.y);
        line.setAttribute("x2", pBot.x);
        line.setAttribute("y2", pBot.y);
        line.setAttribute("stroke", "rgba(255,255,255,0.04)");
        line.setAttribute("stroke-width", "0.6");
        layerGuides.appendChild(line);

        const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
        lbl.setAttribute("x", pTop.x + 3);
        lbl.setAttribute("y", pTop.y + 12);
        lbl.setAttribute("fill", "#4b5563");
        lbl.setAttribute("font-size", "6px");
        lbl.setAttribute("font-family", "var(--font-mono)");
        lbl.textContent = `${{Math.abs(lon).toFixed(1)}}°W`;
        layerGuides.appendChild(lbl);
      }}
      for (let lat = 3.5; lat <= 4.8; lat += 0.25) {{
        const pL = agyProyectarSIG(-74.20, lat, W, H);
        const pR = agyProyectarSIG(-71.02, lat, W, H);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", pL.x);
        line.setAttribute("y1", pL.y);
        line.setAttribute("x2", pR.x);
        line.setAttribute("y2", pR.y);
        line.setAttribute("stroke", "rgba(255,255,255,0.04)");
        line.setAttribute("stroke-width", "0.6");
        layerGuides.appendChild(line);

        const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
        lbl.setAttribute("x", pL.x + 4);
        lbl.setAttribute("y", pL.y - 3);
        lbl.setAttribute("fill", "#4b5563");
        lbl.setAttribute("font-size", "6px");
        lbl.setAttribute("font-family", "var(--font-mono)");
        lbl.textContent = `${{lat.toFixed(2)}}°N`;
        layerGuides.appendChild(lbl);
      }}
    }}

    if (showRivers) {{
      (AGY_DATA.rios_lineas || []).forEach(coords => {{
        const pts = coords.map(c => {{
          const p = agyProyectarSIG(c[0], c[1], W, H);
          return `${{p.x.toFixed(1)}},${{p.y.toFixed(1)}}`;
        }}).join(" ");
        const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        poly.setAttribute("points", pts);
        poly.setAttribute("fill", "none");
        poly.setAttribute("stroke", "#0284c7");
        poly.setAttribute("stroke-width", "0.65");
        poly.setAttribute("stroke-opacity", "0.52");
        layerRivers.appendChild(poly);
      }});
    }}

    if (showField) {{
      (AGY_DATA.puntos_campo || []).forEach(pt => {{
        const p = agyProyectarSIG(pt.longitud, pt.latitud, W, H);
        const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        dot.setAttribute("cx", p.x.toFixed(1));
        dot.setAttribute("cy", p.y.toFixed(1));
        dot.setAttribute("r", "1.9");
        dot.setAttribute("fill", AGY_COLORES_NODO[pt.categoria] || "#9ca3af");
        dot.setAttribute("fill-opacity", "0.78");
        dot.setAttribute("stroke", "#090b0e");
        dot.setAttribute("stroke-width", "0.4");
        dot.style.cursor = "pointer";
        dot.addEventListener("click", (ev) => {{
          ev.stopPropagation();
          agyInspeccionarPuntoCampo(pt);
        }});
        layerField.appendChild(dot);
      }});
      // Etiquetas cartográficas de referencia (Ríos y Centros Urbanos)
      const hitosCartograficos = [
        {{ lon: -72.01, lat: 4.335, txt: "RÍO META", col: "#38bdf8", sz: "3.2px" }},
        {{ lon: -72.135, lat: 4.30, txt: "RÍO MANACACÍAS", col: "#38bdf8", sz: "3.0px" }},
        {{ lon: -71.54, lat: 4.18, txt: "RÍO ITEVIARE", col: "#38bdf8", sz: "3.0px" }},
        {{ lon: -72.955, lat: 4.05, txt: "PUERTO LÓPEZ", col: "#9ca3af", sz: "3.2px" }},
        {{ lon: -73.631, lat: 4.10, txt: "VILLAVICENCIO", col: "#9ca3af", sz: "3.2px" }},
        {{ lon: -74.065, lat: 4.62, txt: "BOGOTÁ D.C.", col: "#f3f4f6", sz: "3.6px" }}
      ];
      hitosCartograficos.forEach(h => {{
        const ph = agyProyectarSIG(h.lon, h.lat, W, H);
        const th = document.createElementNS("http://www.w3.org/2000/svg", "text");
        th.setAttribute("x", ph.x.toFixed(1));
        th.setAttribute("y", ph.y.toFixed(1));
        th.setAttribute("text-anchor", "middle");
        th.setAttribute("fill", h.col);
        th.setAttribute("fill-opacity", "0.75");
        th.setAttribute("font-family", "var(--font-mono)");
        th.setAttribute("font-size", h.sz);
        th.setAttribute("letter-spacing", "0.08em");
        th.setAttribute("pointer-events", "none");
        th.textContent = h.txt;
        layerGuides.appendChild(th);
      }});
    }}
  }} else if (agyLayoutModo === "hier") {{
    const padX = 82;
    AGY_ETAPAS_JERARQUICAS.forEach(et => {{
      const x = padX + (et.col / 5) * (W - 2 * padX);
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", x);
      line.setAttribute("y1", 92);
      line.setAttribute("x2", x);
      line.setAttribute("y2", H - 52);
      line.setAttribute("stroke", "rgba(255,255,255,0.04)");
      line.setAttribute("stroke-width", "1");
      line.setAttribute("stroke-dasharray", "4,4");
      layerGuides.appendChild(line);

      const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
      lbl.setAttribute("x", x);
      lbl.setAttribute("y", 82);
      lbl.setAttribute("text-anchor", "middle");
      lbl.setAttribute("fill", "#9ca3af");
      lbl.setAttribute("font-size", "10px");
      lbl.setAttribute("font-family", "var(--font-mono)");
      lbl.setAttribute("letter-spacing", "0.05em");
      lbl.textContent = et.titulo;
      layerGuides.appendChild(lbl);
    }});
  }} else {{
    const padX = 82, padYTop = 116;
    const zonasGeo = [
      {{ x0: 0.01, x1: 0.21, titulo: "OCCIDENTE · BOGOTÁ / VILLAVICENCIO / PTO. LÓPEZ" }},
      {{ x0: 0.23, x1: 0.87, titulo: "LUPA FOCAL AMPLIADA · PUERTO GAITÁN, RUTA META Y MANACACÍAS" }},
      {{ x0: 0.89, x1: 0.99, titulo: "ORIENTE · PTO. TRUJILLO" }}
    ];
    zonasGeo.forEach((z, idx) => {{
      const zx = padX + z.x0 * (W - 2 * padX);
      const zw = (z.x1 - z.x0) * (W - 2 * padX);
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", zx);
      rect.setAttribute("y", 90);
      rect.setAttribute("width", zw);
      rect.setAttribute("height", H - 144);
      rect.setAttribute("rx", "8");
      rect.setAttribute("fill", idx === 1 ? "rgba(56, 189, 248, 0.02)" : "rgba(255, 255, 255, 0.008)");
      rect.setAttribute("stroke", idx === 1 ? "rgba(56, 189, 248, 0.12)" : "rgba(255, 255, 255, 0.04)");
      rect.setAttribute("stroke-width", "1");
      rect.setAttribute("stroke-dasharray", idx === 1 ? "none" : "4,4");
      layerGuides.appendChild(rect);

      const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
      lbl.setAttribute("x", zx + zw / 2);
      lbl.setAttribute("y", 82);
      lbl.setAttribute("text-anchor", "middle");
      lbl.setAttribute("fill", idx === 1 ? "#38bdf8" : "#9ca3af");
      lbl.setAttribute("font-size", "10px");
      lbl.setAttribute("font-family", "var(--font-mono)");
      lbl.setAttribute("letter-spacing", "0.05em");
      lbl.textContent = z.titulo;
      layerGuides.appendChild(lbl);
    }});

    if (agySlideActual === 2 || agySlideActual === 3) {{
      (AGY_DATA.rios_lineas || []).forEach(coords => {{
        const pts = coords.map(c => {{
          const p = agyProyectarGeoLupa(c[0], c[1], W, H, padX, padYTop);
          return `${{p.x.toFixed(1)}},${{p.y.toFixed(1)}}`;
        }}).join(" ");
        const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        poly.setAttribute("points", pts);
        poly.setAttribute("fill", "none");
        poly.setAttribute("stroke", "#0284c7");
        poly.setAttribute("stroke-width", agySlideActual === 2 ? "1.4" : "0.9");
        poly.setAttribute("stroke-opacity", agySlideActual === 2 ? "0.30" : "0.15");
        layerRivers.appendChild(poly);
      }});
    }}
  }}

  // 3. Determinar ruta de itinerario resaltada
  const aristasItinerario = new Set();
  const nodosItinerario = new Set();
  if (agyItinerarioActivo) {{
    const it = agyListaItinerariosActiva().find(x => x.origen_id === agyItinerarioActivo);
    if (it) {{
      (it.ruta || []).forEach(nid => nodosItinerario.add(nid));
      for (let i = 0; i < it.ruta.length - 1; i++) {{
        aristasItinerario.add(`${{it.ruta[i]}}->${{it.ruta[i + 1]}}`);
      }}
    }}
  }}

  // 4. Dibujar Aristas
  const showRoadsGIS = document.getElementById("chk-layer-roads")?.checked ?? true;
  const showNavRiversGIS = document.getElementById("chk-layer-navrivers")?.checked ?? true;

  aristas.forEach(a => {{
    if (agySlideActual === 0) {{
      if (a.tipo_via === "fluvial" && !showNavRiversGIS) return;
      if (a.tipo_via !== "fluvial" && !showRoadsGIS) return;
    }}

    const p1 = agyPosiciones[a.origen];
    const p2 = agyPosiciones[a.destino];
    if (!p1 || !p2) return;

    const esSeleccionada = agySeleccion && agySeleccion.tipo === "arista" && agySeleccion.id === a.id;
    const esDeItinerario = aristasItinerario.has(a.id);
    const esVecinaHover = agyHoverNodeId && (a.origen === agyHoverNodeId || a.destino === agyHoverNodeId);

    let stroke = "#22c55e";
    let marker = "url(#arrow-terrestre)";
    let dash = "";
    if (a.tipo_via === "fluvial") {{
      stroke = "#38bdf8";
      marker = "url(#arrow-fluvial)";
      dash = agySlideActual === 0 ? "2,1.5" : "6,4";
    }} else if (a.tipo_via === "acceso_local") {{
      stroke = "#9ca3af";
      marker = "url(#arrow-acceso)";
      dash = a.estado_via === "regular" ? (agySlideActual === 0 ? "1.5,1.5" : "4,4") : "";
    }} else {{
      if (a.estado_via === "mala") {{
        stroke = "#ef4444";
        marker = "url(#arrow-mala)";
      }} else if (a.estado_via === "regular") {{
        stroke = "#f97316";
        marker = "url(#arrow-regular)";
      }}
    }}

    if (esSeleccionada || esDeItinerario) {{
      stroke = "#facc15";
      marker = "url(#arrow-active)";
    }}

    let opacidad = "0.84";
    if (agyItinerarioActivo && !esDeItinerario && !esSeleccionada) {{
      opacidad = "0.16";
    }} else if (agyHoverNodeId && !esVecinaHover && !esSeleccionada) {{
      opacidad = "0.20";
    }} else if (esVecinaHover || esSeleccionada || esDeItinerario) {{
      opacidad = "1";
    }}

    let pathD = "";
    let lx = 0, ly = 0;

    if (agySlideActual === 0 && a.geometria_muestra && a.geometria_muestra.length >= 2) {{
      // En Diapositiva 1 (SIG): Trazar la polilínea GPS real (OSRM o cauce de río)
      const ptsProy = a.geometria_muestra.map(pt => agyProyectarSIG(pt[1], pt[0], W, H));
      pathD = ptsProy.map((pt, i) => `${{i === 0 ? 'M' : 'L'}} ${{pt.x.toFixed(1)}} ${{pt.y.toFixed(1)}}`).join(" ");
      const mid = ptsProy[Math.floor(ptsProy.length / 2)];
      lx = mid.x;
      ly = mid.y - 2;
    }} else {{
      const geom = agyRutaCurvaArista(p1, p2);
      pathD = geom.d;
      lx = geom.lx;
      ly = geom.ly;
    }}

    const gEdge = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const pathVis = document.createElementNS("http://www.w3.org/2000/svg", "path");
    pathVis.setAttribute("d", pathD);
    pathVis.setAttribute("class", "agy-edge-path" + (esDeItinerario ? " agy-route-highlight" : ""));
    pathVis.setAttribute("stroke", stroke);

    const baseWidth = agySlideActual === 0
      ? ((esSeleccionada || esDeItinerario) ? "1.6" : "0.85")
      : ((esSeleccionada || esDeItinerario) ? "3.8" : (esVecinaHover ? "2.8" : "2.0"));
    pathVis.setAttribute("stroke-width", baseWidth);
    pathVis.setAttribute("stroke-opacity", opacidad);
    if (dash || esDeItinerario) {{
      pathVis.setAttribute("stroke-dasharray", esDeItinerario ? (agySlideActual === 0 ? "3,2" : "10,6") : dash);
    }}
    if (agySlideActual !== 0) {{
      pathVis.setAttribute("marker-end", marker);
    }}

    const pathHit = document.createElementNS("http://www.w3.org/2000/svg", "path");
    pathHit.setAttribute("d", pathD);
    pathHit.setAttribute("class", "agy-edge-hit");
    if (agySlideActual === 0) pathHit.setAttribute("stroke-width", "5px");
    pathHit.addEventListener("click", (ev) => {{
      ev.stopPropagation();
      agyInspeccionarArista(a);
    }});

    gEdge.appendChild(pathVis);
    gEdge.appendChild(pathHit);

    let txt = null;
    if (agySlideActual !== 0) {{
      txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
      txt.setAttribute("x", lx.toFixed(1));
      txt.setAttribute("y", ly.toFixed(1));
      txt.setAttribute("text-anchor", "middle");
      txt.setAttribute("fill", (esSeleccionada || esDeItinerario) ? "#fef08a" : "#9ca3af");
      txt.setAttribute("fill-opacity", opacidad);
      txt.setAttribute("font-family", "var(--font-mono)");
      txt.setAttribute("font-size", "9.5px");
      txt.setAttribute("font-weight", (esSeleccionada || esDeItinerario) ? "700" : "500");
      txt.setAttribute("style", "paint-order: stroke; stroke: #090b0e; stroke-width: 3.5px; stroke-linejoin: round;");
      txt.setAttribute("pointer-events", "none");
      txt.textContent = `${{a.distancia_km}}km·${{a.tiempo_efectivo_h.toFixed(2)}}h`;
      gEdge.appendChild(txt);
    }}

    layerEdges.appendChild(gEdge);
    agyDomEdges[a.id] = {{ arista: a, pathVis, pathHit, txt }};
  }});

  // 5. Dibujar Nodos (Vértices)
  const showNodesGIS = document.getElementById("chk-layer-nodes")?.checked ?? true;
  if (agySlideActual === 0 && !showNodesGIS) return;

  nodos.forEach(n => {{
    const pos = agyPosiciones[n.id];
    if (!pos) return;

    const esSeleccionado = agySeleccion && agySeleccion.tipo === "nodo" && agySeleccion.id === n.id;
    const esDeItin = nodosItinerario.has(n.id);
    const color = AGY_COLORES_NODO[n.tipo] || "#64748b";
    const esGIS = (agySlideActual === 0);

    const gNode = document.createElementNS("http://www.w3.org/2000/svg", "g");
    gNode.setAttribute("class", "agy-node-group");
    gNode.setAttribute("transform", `translate(${{pos.x.toFixed(1)}}, ${{pos.y.toFixed(1)}})`);

    // En Diapositiva 4 (Multimodal), indicar con doble anillo los nodos de transferencia modal
    if (!esGIS && (n.tipo === "D" || n.tipo === "H" || esSeleccionado || esDeItin || (agySlideActual === 3 && n.es_multimodal))) {{
      const halo = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      halo.setAttribute("r", esSeleccionado ? "22" : "18");
      halo.setAttribute("fill", n.tipo === "D" ? "rgba(239,68,68,0.14)" : (n.tipo === "H" ? "rgba(16,185,129,0.14)" : "rgba(250,204,21,0.12)"));
      halo.setAttribute("stroke", esSeleccionado ? "#facc15" : (n.tipo === "H" ? "#10b981" : color));
      halo.setAttribute("stroke-width", "1.4");
      if (n.tipo === "H" || (agySlideActual === 3 && n.es_multimodal && !esSeleccionado)) {{
        halo.setAttribute("stroke-dasharray", "3,3");
      }}
      gNode.appendChild(halo);
    }}

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", "agy-node-circle");
    circle.setAttribute("r", esGIS ? (esSeleccionado ? "3.2" : "2.1") : "13");
    circle.setAttribute("fill", esGIS ? color : "#11141a");
    circle.setAttribute("stroke", esSeleccionado ? "#facc15" : (esGIS ? "#090b0e" : color));
    circle.setAttribute("stroke-width", esGIS ? "0.6" : (esSeleccionado ? "3.2" : "2.4"));
    gNode.appendChild(circle);

    const tooltip = document.createElementNS("http://www.w3.org/2000/svg", "title");
    tooltip.textContent = `[${{n.id}}] ${{n.nombre}} (${{n.tipo_descripcion}}) — ${{n.latitud.toFixed(4)}}°N, ${{n.longitud.toFixed(4)}}°W`;
    gNode.appendChild(tooltip);

    if (!esGIS) {{
      const lblId = document.createElementNS("http://www.w3.org/2000/svg", "text");
      lblId.setAttribute("text-anchor", "middle");
      lblId.setAttribute("dy", "3.5");
      lblId.setAttribute("fill", esSeleccionado ? "#facc15" : "#f3f4f6");
      lblId.setAttribute("font-family", "var(--font-mono)");
      lblId.setAttribute("font-size", "9.5px");
      lblId.setAttribute("font-weight", "700");
      lblId.setAttribute("pointer-events", "none");
      lblId.textContent = n.id;
      gNode.appendChild(lblId);
    }}

    const lblName = document.createElementNS("http://www.w3.org/2000/svg", "text");
    lblName.setAttribute("text-anchor", "middle");
    lblName.setAttribute("dy", esGIS ? "-3.4" : "26");
    lblName.setAttribute("fill", esSeleccionado ? "#facc15" : "#e5e7eb");
    lblName.setAttribute("font-family", esGIS ? "var(--font-mono)" : "var(--font-sans)");
    lblName.setAttribute("font-size", esGIS ? "2.6px" : "10.5px");
    lblName.setAttribute("font-weight", esGIS ? "700" : "500");
    lblName.setAttribute("style", esGIS
      ? "paint-order: stroke; stroke: #090b0e; stroke-width: 0.8px; stroke-linejoin: round;"
      : "paint-order: stroke; stroke: #090b0e; stroke-width: 4px; stroke-linejoin: round;");
    lblName.setAttribute("pointer-events", "none");
    lblName.textContent = esGIS ? n.id : agyNombreCompacto(n.nombre);
    gNode.appendChild(lblName);

    gNode.addEventListener("mouseenter", () => {{
      if (!agyDraggingNode && agySlideActual !== 0) {{
        agyHoverNodeId = n.id;
        agyActualizarOpacidadHover();
      }}
    }});
    gNode.addEventListener("mouseleave", () => {{
      if (agyHoverNodeId === n.id) {{
        agyHoverNodeId = null;
        agyActualizarOpacidadHover();
      }}
    }});
    gNode.addEventListener("mousedown", (ev) => {{
      ev.stopPropagation();
      if (agySlideActual === 0) return; // En el mapa SIG las posiciones son coordenadas reales fijas
      if (agySpringAnimId) {{
        cancelAnimationFrame(agySpringAnimId);
        agySpringAnimId = null;
      }}
      agyDraggingNode = n.id;
      agyDragMoved = false;
    }});
    gNode.addEventListener("click", (ev) => {{
      ev.stopPropagation();
      if (!agyDragMoved) {{
        agyInspeccionarNodo(n);
      }}
    }});

    layerNodes.appendChild(gNode);
    agyDomNodes[n.id] = gNode;
  }});
}}

// Actualiza opacidades en hover sin reconstruir el DOM
function agyActualizarOpacidadHover() {{
  Object.keys(agyDomEdges).forEach(edgeId => {{
    const ref = agyDomEdges[edgeId];
    const a = ref.arista;
    const esSeleccionada = agySeleccion && agySeleccion.tipo === "arista" && agySeleccion.id === a.id;
    const esVecinaHover = agyHoverNodeId && (a.origen === agyHoverNodeId || a.destino === agyHoverNodeId);
    let op = "0.84";
    let w = "2.0";
    if (agyHoverNodeId) {{
      op = (esVecinaHover || esSeleccionada) ? "1" : "0.18";
      w = (esVecinaHover || esSeleccionada) ? "3.0" : "1.8";
    }}
    ref.pathVis.setAttribute("stroke-opacity", op);
    ref.pathVis.setAttribute("stroke-width", w);
    if (ref.txt) ref.txt.setAttribute("fill-opacity", op);
  }});
}}

// Vista por defecto del Inspector: Resumen Matemático y Lista de Adyacencia
function agyMostrarResumenSubgrafo() {{
  agySeleccion = null;
  const {{ nodos, aristas }} = agySubgrafoActual();
  document.getElementById("agy-insp-title").textContent = "🔎 Estructura del Grafo en Memoria";
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `${{nodos.length}} V · ${{aristas.length}} E`;
  badge.style.background = "var(--bg-canvas)";
  badge.style.color = "var(--accent)";

  const adjLines = nodos.map(n => {{
    const salientes = aristas.filter(a => a.origen === n.id).map(a => `${{a.destino}}(w=${{a.peso_combinado.toFixed(3)}})`).join(", ");
    return `_adyacencia["${{n.id}}"] -> [${{salientes}}]`;
  }}).join("\\n");

  const chipsNodos = nodos.map(n =>
    `<span class="agy-chip" onclick="agySeleccionarElementoRapido('N:${{n.id}}')" style="border-left:3px solid ${{AGY_COLORES_NODO[n.tipo]}}">${{n.id}} · ${{n.nombre}}</span>`
  ).join("");

  const chipsAristas = aristas.slice(0, 16).map(a =>
    `<span class="agy-chip" onclick="agySeleccionarElementoRapido('E:${{a.id}}')">${{a.origen}} → ${{a.destino}} (${{a.tipo_via}})</span>`
  ).join("");

  const flp = AGY_DATA.flp || {{}};

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <b>Exploración Interactiva:</b> Haz clic sobre cualquier vértice o arista en el lienzo para consultar sus atributos reales, grados topológicos y desglose multicriterio.
    </div>

    <div class="agy-prop-group">
      <h4><span>📐 Parámetros del Subgrafo Activo</span><span>G=(V,E)</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Vértices activos |V|</span><span class="agy-prop-val">${{nodos.length}} nodos</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Aristas dirigidas |E|</span><span class="agy-prop-val">${{aristas.length}} arcos</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Estructura base</span><span class="agy-prop-val">Lista Adyacencia O(|V|+|E|)</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Localización FLP (${{flp.metodo || 'greedy'}})</span><span class="agy-prop-val">Centro óptimo: ${{(flp.centros_abiertos || ['F01']).join(', ')}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Límite térmico (T_max)</span><span class="agy-prop-val">${{AGY_DATA.constantes.TIEMPO_MAX_SIN_FRIO_H}} h sin frío</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>📍 Vértices (${{nodos.length}})</span><span>Clic para inspeccionar</span></h4>
      <div class="agy-chip-list">${{chipsNodos}}</div>
    </div>

    <div class="agy-prop-group">
      <h4><span>🔗 Conexiones Dirigidas</span><span>Muestra activa</span></h4>
      <div class="agy-chip-list">${{chipsAristas}}</div>
    </div>

    <div class="agy-prop-group">
      <h4><span>💻 _adyacencia en Memoria</span><span>Python dict</span></h4>
      <div class="agy-code-block">${{adjLines}}</div>
    </div>
  `;
}}

// Mostrar Comparador de Escenarios Reales (Facility Location & Cold Chain Simulation)
function agyMostrarComparadorEscenarios() {{
  const lista = agyDestinoModo === "PG" ? (AGY_DATA.escenarios_pg || []) : (AGY_DATA.escenarios_bog || []);
  const flp = AGY_DATA.flp || {{}};
  document.getElementById("agy-insp-title").textContent = `📊 Evaluación de Escenarios (${{agyDestinoModo === "PG" ? "Pto. Gaitán" : "Bogotá"}})`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `${{lista.length}} Escenarios`;
  badge.style.color = "#facc15";

  const filas = lista.map(e => `
    <tr>
      <td>${{e.nombre}}</td>
      <td style="text-align:right">${{e.num_centros}}</td>
      <td style="text-align:right">${{e.pct_perdida.toFixed(1)}}%</td>
      <td style="text-align:right">$${{(e.costo_total_cop / 1e6).toFixed(2)}}M</td>
      <td style="text-align:right">$${{e.costo_por_kg_cop.toLocaleString()}}</td>
    </tr>
  `).join("");

  const asigs = Object.entries(flp.asignaciones || {{}}).map(([p, f]) =>
    `<span class="agy-chip" onclick="agySeleccionarElementoRapido('N:${{p}}')">${{p}} ➔ ${{f}}</span>`
  ).join("");

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <b>Facility Location Problem (FLP Greedy):</b><br>
      Centro abierto óptimo: <b>${{(flp.centros_abiertos || []).join(", ")}}</b> · Tiempo máx. acceso: <b>${{flp.tiempo_max_acceso_h}} h</b> · Costo fijo: <b>$${{(flp.costo_total_fijo || 0).toLocaleString()}} COP</b>.
    </div>

    <div class="agy-prop-group">
      <h4><span>⚖️ Comparativa Real de Escenarios</span><span>Destino: ${{agyDestinoModo}}</span></h4>
      <table class="agy-table">
        <thead>
          <tr>
            <th>Escenario</th>
            <th style="text-align:right">CF</th>
            <th style="text-align:right">Pérd.</th>
            <th style="text-align:right">Costo Tot.</th>
            <th style="text-align:right">$/kg</th>
          </tr>
        </thead>
        <tbody>${{filas}}</tbody>
      </table>
    </div>

    <div class="agy-prop-group">
      <h4><span>🎯 Asignaciones Puntos de Pesca → Frío</span><span>7 orígenes</span></h4>
      <div class="agy-chip-list">${{asigs}}</div>
    </div>
  `;
}}

// Inspeccionar Vértice (Nodo)
function agyInspeccionarNodo(n) {{
  agySeleccion = {{ tipo: "nodo", id: n.id }};
  agyRenderizarEscenaCompleta(true);

  document.getElementById("agy-insp-title").textContent = `📍 Vértice ${{n.id}}: ${{n.nombre}}`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `Tipo ${{n.tipo}} · ${{n.tipo_descripcion}}`;
  badge.style.background = "var(--bg-canvas)";
  badge.style.color = AGY_COLORES_NODO[n.tipo] || "#38bdf8";

  const aristasSalientes = AGY_DATA.aristas.filter(a => a.origen === n.id);
  const aristasEntrantes = AGY_DATA.aristas.filter(a => a.destino === n.id);

  const chipsSalientes = aristasSalientes.length
    ? aristasSalientes.map(a => `<span class="agy-chip" onclick="agySeleccionarElementoRapido('E:${{a.id}}')">→ ${{a.destino}} (${{a.tipo_via}}, ${{a.distancia_km}} km, w=${{a.peso_combinado.toFixed(3)}})</span>`).join("")
    : `<span style="color:#6b7280;font-size:11px;">Ninguna (grado de salida = 0)</span>`;

  const chipsEntrantes = aristasEntrantes.length
    ? aristasEntrantes.map(a => `<span class="agy-chip" onclick="agySeleccionarElementoRapido('E:${{a.id}}')">← ${{a.origen}} (${{a.tipo_via}}, ${{a.distancia_km}} km)</span>`).join("")
    : `<span style="color:#6b7280;font-size:11px;">Ninguna (grado de entrada = 0)</span>`;

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
    modos_conectados: n.modos_conectados,
    metadata: n.metadata
  }}, null, 2);

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <b>⚡ Efecto algorítmico en el modelo:</b><br>
      ${{n.efecto_vertice}}
    </div>

    <div class="agy-prop-group">
      <h4><span>1. Identidad y Coordenadas WGS84</span><span>Nodo</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">nodo.id</span><span class="agy-prop-val">"${{n.id}}"</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">nodo.nombre</span><span class="agy-prop-val">${{n.nombre}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">nodo.tipo</span><span class="agy-prop-val">${{n.tipo}} (${{n.tipo_descripcion}})</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">coordenadas (lat, lon)</span><span class="agy-prop-val">${{n.latitud.toFixed(4)}}°N, ${{n.longitud.toFixed(4)}}°W</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">modos_conectados</span><span class="agy-prop-val">${{(n.modos_conectados || []).join(" + ") || "—"}}</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>2. Capacidades, Producción y Costos</span><span>Métricas</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">produccion_kg_dia</span><span class="agy-prop-val">${{n.produccion_kg_dia.toFixed(2)}} kg/día</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">capacidad_frio_kg</span><span class="agy-prop-val">${{n.capacidad_frio_kg.toLocaleString()}} kg</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">costo_fijo</span><span class="agy-prop-val">$${{n.costo_fijo.toLocaleString()}} COP/mes</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">demanda_kg_dia</span><span class="agy-prop-val">${{n.metadata.demanda_kg_dia || 0}} kg/día</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">rio_asociado</span><span class="agy-prop-val">${{n.metadata.rio || "—"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">num_pescadores</span><span class="agy-prop-val">${{n.metadata.pescadores || "—"}}</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>3. Infraestructura y Restricciones</span><span>Flags</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">tiene_energia</span><span class="agy-prop-val">${{n.tiene_energia ? "✅ True" : "❌ False"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">tiene_agua</span><span class="agy-prop-val">${{n.tiene_agua ? "✅ True" : "❌ False"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">tiene_muelle</span><span class="agy-prop-val">${{n.tiene_muelle ? "✅ True" : "❌ False"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">restriccion_ambiental</span><span class="agy-prop-val">${{n.restriccion_ambiental ? "🚫 True (Hábitat H)" : "False"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">es_candidato_frio (FLP)</span><span class="agy-prop-val">${{n.es_candidato_frio ? "✅ Candidato elegible" : "❌ No"}}</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>4. Conectividad en Lista de Adyacencia</span><span>deg⁻ / deg⁺</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Grado de Entrada (deg⁻)</span><span class="agy-prop-val">${{n.grado_entrada}} aristas entrantes</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Grado de Salida (deg⁺)</span><span class="agy-prop-val">${{n.grado_salida}} aristas salientes</span></div>
      <div style="margin-top:8px;color:#9ca3af;font-size:11px;">Aristas salientes desde ${{n.id}}:</div>
      <div class="agy-chip-list">${{chipsSalientes}}</div>
      <div style="margin-top:8px;color:#9ca3af;font-size:11px;">Aristas entrantes hacia ${{n.id}}:</div>
      <div class="agy-chip-list">${{chipsEntrantes}}</div>
    </div>

    <div class="agy-prop-group">
      <h4><span>5. Instancia Serializada</span><span>src/models/nodo.py</span></h4>
      <div class="agy-code-block">${{pythonObj}}</div>
    </div>
  `;
}}

// Inspeccionar Arista (22 variables y desglose multicriterio real)
function agyInspeccionarArista(a) {{
  agySeleccion = {{ tipo: "arista", id: a.id }};
  agyRenderizarEscenaCompleta(true);

  const nO = AGY_DATA.nodos.find(x => x.id === a.origen);
  const nD = AGY_DATA.nodos.find(x => x.id === a.destino);

  document.getElementById("agy-insp-title").textContent = `🔗 Arista #${{String(a.idx).padStart(2, '0')}}: ${{a.origen}} → ${{a.destino}}`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `${{a.tipo_via}} (${{a.estado_via}})`;
  badge.style.background = "var(--bg-canvas)";
  badge.style.color = a.tipo_via === "fluvial" ? "#38bdf8" : (a.tipo_via === "acceso_local" ? "#9ca3af" : "#22c55e");

  const w = a.peso_combinado || 0.0001;
  const d = a.desglose_peso;
  const pct = (val) => Math.min(100, Math.round((val / w) * 100));

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <b>${{nO ? nO.nombre : a.origen}} (${{a.origen}}) ➔ ${{nD ? nD.nombre : a.destino}} (${{a.destino}})</b><br>
      Geometría real almacenada: <b>${{a.num_puntos_geometria}} vértices GPS</b> (${{a.tipo_via === "fluvial" ? "Cauce hidrográfico rios.shp" : "Trazado vial OSRM"}}).
      <div class="agy-chip-list" style="margin-top:8px;">
        <span class="agy-chip" onclick="agySeleccionarElementoRapido('N:${{a.origen}}')">📍 Nodo origen ${{a.origen}}</span>
        <span class="agy-chip" onclick="agySeleccionarElementoRapido('N:${{a.destino}}')">🏁 Nodo destino ${{a.destino}}</span>
      </div>
    </div>

    <div class="agy-prop-group">
      <h4><span>⚖️ Peso Multicriterio</span><span>w_ij = ${{a.peso_combinado.toFixed(4)}}</span></h4>
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
      <h4><span>1. Distancia, Tiempo y Velocidad</span><span>Cinemática</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">distancia_km</span><span class="agy-prop-val">${{a.distancia_km.toFixed(2)}} km</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">tiempo_h (base)</span><span class="agy-prop-val">${{a.tiempo_h.toFixed(2)}} h</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">flujo_vehicular (factor)</span><span class="agy-prop-val">${{a.flujo_vehicular}} (×${{a.factor_flujo}})</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">tiempo_espera_h</span><span class="agy-prop-val">${{a.tiempo_espera_h.toFixed(2)}} h</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">tiempo_efectivo_h</span><span class="agy-prop-val">${{a.tiempo_efectivo_h.toFixed(3)}} h</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">velocidad_efectiva</span><span class="agy-prop-val">${{a.velocidad_efectiva_kmh.toFixed(1)}} km/h</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>2. Costos Logísticos y Peajes</span><span>COP</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">costo_cop (flete base)</span><span class="agy-prop-val">$${{a.costo_cop.toLocaleString()}} COP</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">peajes (casetas)</span><span class="agy-prop-val">${{a.peajes}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">costo_peajes_cop</span><span class="agy-prop-val">$${{a.costo_peajes_cop.toLocaleString()}} COP</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">costo_total_cop</span><span class="agy-prop-val">$${{a.costo_total_cop.toLocaleString()}} COP</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>3. Estado Vial, Riesgo y Deterioro</span><span>22 Variables</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">tipo_via</span><span class="agy-prop-val">${{a.tipo_via}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">estado_via (factor)</span><span class="agy-prop-val">${{a.estado_via}} (×${{a.factor_estado_via}})</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">confiabilidad_via</span><span class="agy-prop-val">${{a.confiabilidad_via.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">prob_retraso</span><span class="agy-prop-val">${{a.prob_retraso.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">riesgo_termico</span><span class="agy-prop-val">${{a.riesgo_termico.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">riesgo_accidente</span><span class="agy-prop-val">${{a.riesgo_accidente.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">riesgo_derrumbe</span><span class="agy-prop-val">${{a.riesgo_derrumbe.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">riesgo_combinado</span><span class="agy-prop-val">${{a.riesgo_combinado.toFixed(4)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">perdida_esperada (L_ij)</span><span class="agy-prop-val">${{a.perdida_esperada.toFixed(2)}}%</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">capacidad_kg / peso_max</span><span class="agy-prop-val">${{a.capacidad_kg}} kg · ${{a.restriccion_peso_ton}} t</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">horario_restriccion</span><span class="agy-prop-val">${{a.horario_restriccion}}</span></div>
    </div>
  `;
}}

// Inspeccionar un Punto de Levantamiento de Campo (49 puntos GeoJSON)
function agyInspeccionarPuntoCampo(pt) {{
  agySeleccion = {{ tipo: "campo", id: pt.id }};
  document.getElementById("agy-insp-title").textContent = `🛰️ Punto de Campo ${{pt.id}}`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = pt.categoria_texto;
  badge.style.color = AGY_COLORES_NODO[pt.categoria] || "#38bdf8";

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <b>Registro de Levantamiento Georreferenciado (puntos_campo.geojson)</b><br>
      Punto verificado en terreno dentro del área de influencia hidrográfica del Meta.
    </div>
    <div class="agy-prop-group">
      <h4><span>Atributos GeoJSON</span><span>Feature</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">id</span><span class="agy-prop-val">${{pt.id}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">tipo</span><span class="agy-prop-val">${{pt.tipo}} (${{pt.categoria}})</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">latitud_decimal</span><span class="agy-prop-val">${{pt.latitud.toFixed(6)}}°N</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">longitud_decimal</span><span class="agy-prop-val">${{pt.longitud.toFixed(6)}}°W</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">lat_dms</span><span class="agy-prop-val">${{pt.lat_dms}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">lon_dms</span><span class="agy-prop-val">${{pt.lon_dms}}</span></div>
    </div>
  `;
}}

function agyInspeccionarItinerario(it) {{
  document.getElementById("agy-insp-title").textContent = `🐟 Itinerario ${{it.origen_id}} → ${{it.destino_id}}`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `${{it.distancia_total_km}} km · ${{it.tiempo_total_h}} h`;
  badge.style.background = "var(--bg-canvas)";
  badge.style.color = "#facc15";

  const filasTramos = (it.tramos || []).map((t, i) => `
    <div class="agy-prop-row" style="cursor:pointer" onclick="agySeleccionarElementoRapido('E:${{t.origen}}->${{t.destino}}')">
      <span class="agy-prop-key">${{i+1}}. ${{t.origen}} → ${{t.destino}} (${{t.tipo_via}})</span>
      <span class="agy-prop-val">${{t.distancia_km}} km · ${{t.tiempo_h}} h ${{t.efecto ? '· ' + t.efecto : ''}}</span>
    </div>
  `).join("");

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <b>Secuencia Multimodal Óptima (ADR-06):</b><br>
      ${{it.ruta.join(" ➔ ")}}<br>
      ${{it.nodos_restringidos_evitados.length ? `🌿 <b>Hábitats protegidos evitados:</b> ${{it.nodos_restringidos_evitados.join(", ")}}` : "✅ Ruta libre de restricciones ambientales."}}
    </div>
    <div class="agy-prop-group">
      <h4><span>📊 Métricas Consolidadas</span><span>Dijkstra + ColdChain</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Acopio obligatorio</span><span class="agy-prop-val">${{it.acopio_id || "Directo urbano (P06)"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Centro de Frío</span><span class="agy-prop-val">${{it.centro_frio_id}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Distancia Total</span><span class="agy-prop-val">${{it.distancia_total_km}} km</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Tiempo Total</span><span class="agy-prop-val">${{it.tiempo_total_h}} h</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Carga Inicial → Final</span><span class="agy-prop-val">${{it.kg_inicial}} kg → ${{it.kg_final}} kg</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Pérdida Térmica</span><span class="agy-prop-val">${{it.perdida_pct}}%</span></div>
    </div>
    <div class="agy-prop-group">
      <h4><span>🛣️ Tramos del Recorrido</span><span>Clic para inspeccionar</span></h4>
      ${{filasTramos}}
    </div>
  `;
}}

// Eventos de interacción sobre el lienzo SVG (Arrastre, Paneo, Zoom centrado en cursor y Teclado)
(function initEventosPlataforma() {{
  agyPoblarSelectorItinerarios();
  const svg = document.getElementById("agy-graph-svg");

  svg.addEventListener("mousedown", (ev) => {{
    if (agyDraggingNode) return;
    agyPanning = true;
    agyPanStart = {{ x: ev.clientX - agyTransform.x, y: ev.clientY - agyTransform.y }};
  }});

  window.addEventListener("mousemove", (ev) => {{
    const rect = svg.getBoundingClientRect();
    const sx = (ev.clientX - rect.left - agyTransform.x) / agyTransform.k;
    const sy = (ev.clientY - rect.top - agyTransform.y) / agyTransform.k;

    if (agySlideActual === 0) {{
      const coords = agyInvertirSIG(sx, sy, svg.clientWidth || 960, svg.clientHeight || 640);
      const hudPos = document.getElementById("agy-hud-pos");
      if (hudPos && coords.lat >= 2.5 && coords.lat <= 6.0 && coords.lon >= -75.5 && coords.lon <= -70.0) {{
        hudPos.textContent = `LAT ${{coords.lat.toFixed(4)}}°N · LON ${{Math.abs(coords.lon).toFixed(4)}}°W`;
      }}
    }}

    if (agyDraggingNode) {{
      agyDragMoved = true;
      agyPosiciones[agyDraggingNode] = {{ x: sx, y: sy }};
      if (agyTargets[agyDraggingNode]) {{
        agyTargets[agyDraggingNode] = {{ x: sx, y: sy }};
      }}
      agyActualizarGeometriaDOM();
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
    const rect = svg.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    const factor = ev.deltaY < 0 ? 1.15 : 0.87;
    const nuevoK = Math.min(18.0, Math.max(0.45, agyTransform.k * factor));
    const ratio = nuevoK / agyTransform.k;
    agyTransform.x = mx - (mx - agyTransform.x) * ratio;
    agyTransform.y = my - (my - agyTransform.y) * ratio;
    agyTransform.k = nuevoK;
    agyActualizarTransform();
  }}, {{ passive: false }});

  // Navegación instantánea por teclado (100+/día rule -> viaKeyboard = true, 0ms animation)
  window.addEventListener("keydown", (ev) => {{
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (ev.key === "ArrowRight") {{
      agyCambiarSlide(agySlideActual + 1, true);
    }} else if (ev.key === "ArrowLeft") {{
      agyCambiarSlide(agySlideActual - 1, true);
    }} else if (["1", "2", "3", "4"].includes(ev.key)) {{
      agyCambiarSlide(Number(ev.key) - 1, true);
    }} else if (ev.key === "r" || ev.key === "R") {{
      agyResetVista();
    }} else if (ev.key === "Escape") {{
      agyItinerarioActivo = "";
      document.getElementById("agy-itin-select").value = "";
      agyRenderizarEscenaCompleta(true);
      agyMostrarResumenSubgrafo();
    }}
  }});

  window.addEventListener("resize", () => {{
    if (agySlideActual > 0 || AGY_STANDALONE) {{
      agyRenderizarEscenaCompleta(true);
    }}
  }});

  if (AGY_STANDALONE) {{
    setTimeout(() => {{
      agyCambiarSlide(0, true);
    }}, 20);
  }}
}})();
</script>
"""


def generar_html_slides(
    grafo: GrafoLogistico,
    geo_cache: Optional[Dict[str, list]] = None,
    itinerarios: Optional[list] = None,
) -> str:
    """
    Devuelve el bloque HTML + CSS + JS para inyectar sobre `folium_map.py`.
    """
    payload = construir_datos_slides(grafo, geo_cache, itinerarios)
    json_payload = json.dumps(payload, ensure_ascii=False)
    return _construir_bloque_ui_interactiva(json_payload, modo_autonomo=False)


def generar_presentacion_html_autonoma(
    grafo: GrafoLogistico,
    geo_cache: Optional[Dict[str, list]] = None,
    itinerarios: Optional[list] = None,
    ruta_salida: Optional[str] = None,
) -> str:
    """
    Genera el archivo HTML autónomo y 100% offline en:
    `data/output/presentacion_red_logistica.html`.
    Crea automáticamente una copia de seguridad (`.backup.html`) si el archivo ya existe.
    """
    if ruta_salida is None:
        ruta_path = OUTPUT_DIR / "presentacion_red_logistica.html"
    else:
        ruta_path = Path(ruta_salida)

    ruta_path.parent.mkdir(parents=True, exist_ok=True)
    if ruta_path.exists():
        backup_path = ruta_path.with_suffix(".backup.html")
        shutil.copy2(ruta_path, backup_path)

    payload = construir_datos_slides(grafo, geo_cache, itinerarios)
    json_payload = json.dumps(payload, ensure_ascii=False)
    cuerpo = _construir_bloque_ui_interactiva(json_payload, modo_autonomo=True)

    html_completo = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Presentación Interactiva — Red Logística Multimodal Puerto Gaitán</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%2311141a'/><text x='16' y='21' font-family='monospace' font-size='14' font-weight='bold' fill='%2338bdf8' text-anchor='middle'>G</text></svg>">
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #090b0e;
      overflow: hidden;
    }}
  </style>
</head>
<body>
{cuerpo}
</body>
</html>
"""
    with open(ruta_path, "w", encoding="utf-8") as f:
        f.write(html_completo)
    print(f"  [✓] Presentación autónoma offline generada: {ruta_path}")
    return str(ruta_path)
