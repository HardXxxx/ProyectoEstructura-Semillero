"""
graph_slides.py — Experiencia Interactiva de Visualización Logística y Presentación Magistral
para la Red Multimodal de Pesca Artesanal (Puerto Gaitán, Meta → Villavicencio / Bogotá).

Fusión de dirección de arte y sistemas de diseño:
  • Dala ("Constellation floating on black velvet"): Lienzo de grafos sobre negro absoluto
    (#000000), constelación reactiva de partículas triangulares multicromáticas (#8052ff,
    #ffb829, #15846e, #2997ff), botones píldora en Electric Iris (#8052ff, radio 24px) y
    tipografía escultural PPNeueMontreal de peso 400 con tracking negativo (-0.035em) y
    cuerpo ligero (peso 300).
  • Apple ("Midnight Hardware Gallery"): Módulos de exhibición en Obsidian (#000000) y
    Carbon (#1d1d1f) con radio de 28px, contornos de 1px (#333336), cero sombras difusas,
    tipografía Porcelain (#f5f5f7) / Ash (#86868b) y píldoras de acción Apple Blue (#0071e3).
  • ORYZO AI ("Darkroom Product Editorial"): Contraste editorial en Warm Cream (#ffedd7),
    separadores estructurales discontinuos (1px dashed #40372e), acentos Ember (#dc5000) y
    etiqueta serial vertical de ingeniería en el margen derecho.
  • Emil Kowalski (`animate`, `animation-performance`, `choreograph`, `emil-design-eng`) +
    Impeccable (`impeccable`, `impeccable-design`, `craft-floor.md`) + Taste (`taste.md`).
"""

from __future__ import annotations

import json
import shutil
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..models.grafo_logistico import GrafoLogistico
from ..algorithms.dijkstra import dijkstra
from ..algorithms.bellman_ford import bellman_ford
from ..algorithms.floyd_warshall import floyd_warshall, reconstruir_camino
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


def _calcular_bfs_saltos(grafo: GrafoLogistico, destino_frio: str = "F02") -> List[Dict[str, Any]]:
    """
    Implementación propia de Búsqueda en Amplitud (BFS) con cola FIFO (collections.deque)
    en O(|V| + |E|) para hallar el camino con menor número de saltos/transbordos desde cada
    punto de pesca hacia el centro de frío principal, evitando hábitats restringidos (H).
    """
    resultados = []
    restringidos = {n.id for n in grafo.nodos if n.restriccion_ambiental}

    for p in sorted(grafo.puntos_pesca(), key=lambda x: x.id):
        cola = deque([(p.id, [p.id])])
        visitados = {p.id}
        camino_hallado: List[str] = []

        while cola:
            actual, camino = cola.popleft()
            if actual == destino_frio:
                camino_hallado = camino
                break
            for ar in grafo.obtener_aristas(actual):
                vec = ar.destino
                if vec in visitados:
                    continue
                if vec in restringidos and vec != destino_frio:
                    continue
                visitados.add(vec)
                cola.append((vec, camino + [vec]))

        resultados.append({
            "origen_id": p.id,
            "nombre": p.nombre,
            "destino_id": destino_frio,
            "saltos": max(0, len(camino_hallado) - 1),
            "camino": camino_hallado,
        })
    return resultados


def _calcular_dfs_rutas_alternativas(
    grafo: GrafoLogistico, destino_frio: str = "F02", max_profundidad: int = 7
) -> List[Dict[str, Any]]:
    """
    Implementación propia de Búsqueda en Profundidad (DFS) con retroceso (backtracking)
    para enumerar rutas simples alternativas libres de restricción ambiental entre cada
    punto de captura y el centro de frío.
    """
    restringidos = {n.id for n in grafo.nodos if n.restriccion_ambiental}
    resumen_dfs = []

    for p in sorted(grafo.puntos_pesca(), key=lambda x: x.id):
        rutas_simples: List[Dict[str, Any]] = []

        def _backtrack(actual: str, camino: List[str], tiempo_acum: float, dist_acum: float) -> None:
            if len(camino) - 1 > max_profundidad:
                return
            if actual == destino_frio:
                rutas_simples.append({
                    "camino": list(camino),
                    "saltos": len(camino) - 1,
                    "tiempo_h": round(tiempo_acum, 2),
                    "distancia_km": round(dist_acum, 2),
                })
                return
            for ar in grafo.obtener_aristas(actual):
                vec = ar.destino
                if vec in camino:
                    continue
                if vec in restringidos and vec != destino_frio:
                    continue
                _backtrack(
                    vec,
                    camino + [vec],
                    tiempo_acum + ar.tiempo_h,
                    dist_acum + ar.distancia_km,
                )

        _backtrack(p.id, [p.id], 0.0, 0.0)
        rutas_simples.sort(key=lambda r: (r["tiempo_h"], r["saltos"]))
        resumen_dfs.append({
            "origen_id": p.id,
            "nombre": p.nombre,
            "num_rutas": len(rutas_simples),
            "mejor_ruta": rutas_simples[0]["camino"] if rutas_simples else [],
            "ruta_contingencia": rutas_simples[1]["camino"] if len(rutas_simples) > 1 else (
                rutas_simples[0]["camino"] if rutas_simples else []
            ),
            "rutas": rutas_simples[:4],
        })
    return resumen_dfs


def construir_datos_slides(
    grafo: GrafoLogistico,
    geo_cache: Optional[Dict[str, list]] = None,
    itinerarios: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Extrae y serializa todos los datos reales del repositorio (nodos, aristas, geometrías OSRM,
    red hidrográfica rios.shp, 49 puntos de campo, itinerarios, Dijkstra, Bellman-Ford,
    Floyd-Warshall, BFS, DFS y evaluación de escenarios FLP) sin inventar valores.
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
                f"Nodo de transferencia modal (fluvial -> terrestre): añade +{TIEMPO_TRANSBORDO_MUELLE_H:.2f} h "
                "al tiempo acumulado sin frío y contabiliza +1 operación de transbordo bajo radiación solar."
            )
        elif n.tipo == "D":
            d["efecto_vertice"] = (
                f"Zona crítica de deterioro térmico: multiplica por x{FACTOR_DETERIORO_ZONA_D} "
                "la tasa de degradación del tramo entrante (de 2.5%/h a 3.75%/h)."
            )
        elif n.tipo == "H":
            d["efecto_vertice"] = (
                "Hábitat protegido de desove (restriccion_ambiental=True): bloqueado como vértice "
                "intermedio en Dijkstra, Bellman-Ford, Floyd-Warshall, BFS y DFS."
            )
        elif n.tipo == "F":
            d["efecto_vertice"] = (
                f"Planta frigorífica ({n.capacidad_frio_kg:,.0f} kg, costo fijo ${n.costo_fijo:,.0f} COP/mes): "
                "abate la temperatura a 0 °C (hielo 1:1) y reinicia tiempo_sin_frio = 0.0 h."
            )
        elif n.tipo == "A":
            d["efecto_vertice"] = (
                f"Centro de acopio obligatorio (ADR-06, capacidad {n.capacidad_frio_kg:,.0f} kg): "
                + ("dispone de energía eléctrica continua y pre-frío."
                   if n.tiene_energia else "sin energía eléctrica confiable (excluido como candidato FLP).")
            )
        elif n.tipo == "P":
            d["efecto_vertice"] = (
                f"Asociación pesquera artesanal ({n.metadata.get('rio', 'Cuenca Meta')}): "
                f"captura censada de {n.produccion_kg_dia:.2f} kg/día con {n.metadata.get('pescadores', 0)} pescadores."
            )
        else:
            d["efecto_vertice"] = (
                f"Mercado comercial mayorista/minorista (demanda diaria: {n.metadata.get('demanda_kg_dia', 0)} kg/día)."
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

    # 5. Tiempos de acceso a cadena de frío por asociación pesquera (P01..P07 -> F02)
    tiempos_acceso_frio = []
    for p in sorted(grafo.puntos_pesca(), key=lambda x: x.id):
        t_frio, cam_frio = dijkstra(grafo, p.id, "F02", criterio="tiempo_h", evitar_restringidos=True)
        dist_frio = 0.0
        for i in range(len(cam_frio) - 1):
            ar_tr = grafo.obtener_arista(cam_frio[i], cam_frio[i + 1])
            if ar_tr:
                dist_frio += ar_tr.distancia_km
        minutos = round(t_frio * 60) if t_frio < float("inf") else 0
        tiempos_acceso_frio.append({
            "origen_id": p.id,
            "nombre": p.nombre,
            "rio": p.metadata.get("rio", "Meta"),
            "produccion_kg_dia": round(p.produccion_kg_dia, 2),
            "pescadores": p.metadata.get("pescadores", 0),
            "centro_frio": "F02",
            "tiempo_h": round(t_frio, 2) if t_frio < float("inf") else 0.0,
            "tiempo_min": minutos,
            "distancia_km": round(dist_frio, 2),
            "camino_frio": cam_frio,
            "cumple_fao": bool(t_frio <= TIEMPO_MAX_SIN_FRIO_H),
            "meta_excelente": bool(t_frio <= 1.0),
        })

    # 6. Verificación cruzada Bellman-Ford vs Dijkstra y Matriz Floyd-Warshall
    verificacion_bf = []
    tiene_ciclo_negativo = False
    try:
        dist_bf, _, tiene_ciclo_negativo = bellman_ford(
            grafo, "P01", criterio="tiempo_h", evitar_restringidos=True
        )
        destinos_clave = ["M03", "A01", "A02", "F01", "F02", "F03", "V_PG1", "V_VLL", "V_BOG1"]
        for d_id in destinos_clave:
            if d_id in grafo:
                t_dij, cam_dij = dijkstra(grafo, "P01", d_id, criterio="tiempo_h", evitar_restringidos=True)
                t_bf = dist_bf.get(d_id, float("inf"))
                if t_dij < float("inf") and t_bf < float("inf"):
                    verificacion_bf.append({
                        "origen": "P01",
                        "destino": d_id,
                        "nombre_destino": grafo.obtener_nodo(d_id).nombre if grafo.obtener_nodo(d_id) else d_id,
                        "dijkstra_h": round(t_dij, 4),
                        "bellman_ford_h": round(t_bf, 4),
                        "diferencia": round(abs(t_dij - t_bf), 6),
                        "coincide": abs(t_dij - t_bf) < 1e-5,
                        "camino": cam_dij,
                    })
    except Exception:
        pass

    matriz_fw_muestra = []
    try:
        dist_fw, sig_fw = floyd_warshall(grafo, criterio="tiempo_h", evitar_restringidos=True)
        cols_fw = ["A01", "A02", "M03", "F01", "F02", "F03", "V_PG1", "V_BOG1"]
        for p in sorted(grafo.puntos_pesca(), key=lambda x: x.id):
            fila_vals = {}
            for c_id in cols_fw:
                val = dist_fw.get(p.id, {}).get(c_id, float("inf"))
                cam = reconstruir_camino(sig_fw, p.id, c_id) if val < float("inf") else []
                fila_vals[c_id] = {
                    "tiempo_h": round(val, 2) if val < float("inf") else None,
                    "camino": cam,
                }
            matriz_fw_muestra.append({
                "origen_id": p.id,
                "nombre": p.nombre,
                "destinos": fila_vals,
            })
    except Exception:
        pass

    bfs_saltos = _calcular_bfs_saltos(grafo, "F02")
    dfs_rutas = _calcular_dfs_rutas_alternativas(grafo, "F02")

    # 7. Resultados reales de FLP y Evaluador de Escenarios
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
            {"nombre": "3 Centros (F01+F02+F03)", "centros": ["F01", "F02", "F03"]},
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

    # 8. Red Fluvial Topológica (rios.shp) y 49 Puntos de Campo (puntos_campo.geojson)
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
        "tiempos_acceso_frio": tiempos_acceso_frio,
        "verificacion_bf": verificacion_bf,
        "tiene_ciclo_negativo": tiene_ciclo_negativo,
        "matriz_fw": matriz_fw_muestra,
        "bfs_saltos": bfs_saltos,
        "dfs_rutas": dfs_rutas,
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
    Construye el sistema completo de 8 diapositivas interactivas combinando:
      - Estética Dala ("Constellation floating on black velvet") para el lienzo interactivo de
        grafos, nodos, aristas y partículas triangulares multicromáticas sobre #000000.
      - Estética Apple ("Midnight Hardware Gallery") + ORYZO AI ("Darkroom Product Editorial")
        para las diapositivas de exposición ejecutiva, modelo matemático, solución FLP e impacto.
    """
    clase_slide0 = "visible" if modo_autonomo else "visible"
    return f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Albert+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400;1,6..72,500&display=swap');

  :root {{
    /* ═══════════════════════════════════════════════════════════════
       TOKENS CROMÁTICOS: DALA × APPLE MIDNIGHT × ORYZO EDITORIAL
       ═══════════════════════════════════════════════════════════════ */
    --color-void: #000000;
    --color-obsidian: #000000;
    --color-graphite: #0e0e0e;
    --color-carbon: #1d1d1f;
    --color-walnut-shadow: #100904;
    --color-bark-brown: #382416;
    --color-steel: #333336;
    --color-cork-border: #40372e;
    --color-slate: #6e6e73;

    /* Tipografía de alto contraste */
    --color-bone-white: #ffffff;
    --color-porcelain: #f5f5f7;
    --color-warm-cream: #ffedd7;
    --color-silver-mist: #c8c8cd;
    --color-ash-gray: #9a9a9a;
    --color-ash: #8e8e93;

    /* Acentos de marca e ingeniería (calibrados a WCAG AA >= 4.5:1) */
    --color-electric-iris: #6836f5;
    --color-saffron-spark: #ffb829;
    --color-deep-verdant: #15846e;
    --color-apple-blue: #0062c4;
    --color-electric-link: #2997ff;
    --color-optical-yellow: #ffd500;
    --color-ember-accent: #ff6b2b;

    /* Colores semánticos de los 7 tipos de vértices G=(V,E) */
    --col-pesca: #2997ff;
    --col-muelle: #8052ff;
    --col-acopio: #ffb829;
    --col-frio: #15846e;
    --col-deterioro: #dc5000;
    --col-habitat: #10b981;
    --col-venta: #ec4899;

    /* Familias tipográficas refinadas */
    --font-sans: 'PPNeueMontreal', 'Albert Sans', 'SF Pro Display', 'Segoe UI Variable Display', 'Segoe UI', system-ui, -apple-system, sans-serif;
    --font-serif: 'Newsreader', 'Georgia', 'Times New Roman', serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', 'Cascadia Code', Consolas, monospace;

    /* Radios y geometría (Dala 24px pills / Apple 28px plinths) */
    --radius-cards: 28px;
    --radius-panel: 24px;
    --radius-nav: 20px;
    --radius-pill: 9999px;

    /* Curvas Emil Kowalski (animate & review-animations) */
    --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
    --ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
    --ease-drawer: cubic-bezier(0.32, 0.72, 0, 1);
  }}

  *, *::before, *::after {{
    box-sizing: border-box;
  }}

  /* Integración limpia con el mapa Folium original en Diapositiva 2 */
  body:not(.folium-embed-only) .leaflet-top {{
    top: 66px !important;
  }}
  body:not(.folium-embed-only) #folium-title-card {{
    top: 68px !important;
  }}
  body:not(.folium-embed-only) #folium-itin-card {{
    top: 138px !important;
  }}
  body.folium-embed-only #agy-slide-nav,
  body.folium-embed-only #agy-app-shell,
  body.folium-embed-only #agy-oryzo-serial {{
    display: none !important;
  }}

  /* Superficies del navegador personalizadas (craft-floor.md) */
  ::selection {{
    background: rgba(128, 82, 255, 0.38);
    color: var(--color-bone-white);
  }}
  ::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
  }}
  ::-webkit-scrollbar-track {{
    background: var(--color-void);
  }}
  ::-webkit-scrollbar-thumb {{
    background: var(--color-steel);
    border-radius: 9999px;
  }}
  ::-webkit-scrollbar-thumb:hover {{
    background: var(--color-electric-iris);
  }}
  :focus-visible {{
    outline: 2px solid var(--color-electric-iris);
    outline-offset: 2px;
  }}

  /* Escala tipográfica jerárquica (pasos > 1.35× con legibilidad editorial) */
  body {{
    font-family: var(--font-sans);
    font-size: 16px;
    line-height: 1.55;
    color: var(--color-porcelain);
    background-color: var(--color-void);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }}
  h1 {{
    font-family: var(--font-sans);
    font-size: 46px;
    font-weight: 500;
    line-height: 1.08;
    letter-spacing: -0.026em;
    color: var(--color-bone-white);
    margin: 0;
  }}
  h2 {{
    font-family: var(--font-sans);
    font-size: 30px;
    font-weight: 500;
    line-height: 1.15;
    letter-spacing: -0.02em;
    color: var(--color-porcelain);
    margin: 0;
  }}
  h3 {{
    font-family: var(--font-sans);
    font-size: 20px;
    font-weight: 600;
    line-height: 1.25;
    letter-spacing: -0.012em;
    color: var(--color-porcelain);
    margin: 0;
  }}
  p {{
    font-size: 16px;
    font-weight: 400;
    line-height: 1.6;
    color: var(--color-silver-mist);
    margin: 0;
  }}

  /* Números tabulares obligatorios en métricas, coordenadas, pesos y contadores */
  .tabular, .agy-kpi-val, .agy-prop-val, .agy-bar-label, .agy-code-block,
  #agy-slide-counter, #agy-coords-hud, .agy-table td, .agy-table th, .agy-stat-num {{
    font-variant-numeric: tabular-nums;
  }}

  /* Etiqueta vertical serial ORYZO en el margen derecho */
  #agy-oryzo-serial {{
    position: fixed;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    writing-mode: vertical-rl;
    text-orientation: mixed;
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--color-ash-gray);
    pointer-events: none;
    z-index: 9800;
    white-space: nowrap;
  }}

  /* ═══════════════════════════════════════════════════════════════
     BARRA GLOBAL DE NAVEGACIÓN (60px = 10 × 6px Dala unit)
     ═══════════════════════════════════════════════════════════════ */
  #agy-slide-nav {{
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: rgba(0, 0, 0, 0.88);
    backdrop-filter: blur(16px) saturate(180%);
    color: var(--color-bone-white);
    padding: 0 24px;
    border-bottom: 1px solid var(--color-steel);
    font-family: var(--font-sans);
    user-select: none;
  }}

  .agy-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    min-width: 0;
    flex-shrink: 0;
  }}
  .agy-brand-mark {{
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: rgba(128, 82, 255, 0.16);
    border: 1px solid rgba(128, 82, 255, 0.48);
    color: var(--color-electric-iris);
    display: grid;
    place-items: center;
    flex-shrink: 0;
  }}
  .agy-brand-text {{
    display: flex;
    flex-direction: column;
    min-width: 0;
  }}
  .agy-brand-title {{
    font-size: 14px;
    line-height: 20px;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: var(--color-warm-cream);
    white-space: nowrap;
  }}
  .agy-brand-sub {{
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
    color: var(--color-ash-gray);
    white-space: nowrap;
  }}

  /* Pestañas de las 8 Diapositivas */
  .agy-slide-tabs {{
    display: flex;
    align-items: center;
    gap: 3px;
    background: var(--color-graphite);
    padding: 4px;
    border-radius: var(--radius-nav);
    border: 1px solid var(--color-steel);
    overflow-x: auto;
    scrollbar-width: none;
    max-width: 100%;
  }}
  .agy-slide-tabs::-webkit-scrollbar {{
    display: none;
  }}

  .agy-slide-tab {{
    background: transparent;
    color: var(--color-ash-gray);
    border: 1px solid transparent;
    padding: 5px 10px;
    border-radius: 16px;
    font-family: var(--font-sans);
    font-size: 12px;
    line-height: 16px;
    font-weight: 600;
    letter-spacing: 0.015em;
    text-transform: uppercase;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    transition: transform 150ms var(--ease-out),
                color 150ms var(--ease-out),
                background-color 150ms var(--ease-out),
                border-color 150ms var(--ease-out);
    white-space: nowrap;
  }}
  .agy-slide-tab:active {{
    transform: scale(0.97);
  }}
  .agy-slide-tab.active {{
    background: var(--color-electric-iris);
    color: var(--color-bone-white);
    border-color: rgba(255, 255, 255, 0.18);
  }}
  .agy-slide-tab.active.is-editorial {{
    background: var(--color-apple-blue);
  }}
  .agy-tab-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    opacity: 0.7;
  }}

  .agy-nav-actions {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }}

  #agy-slide-counter {{
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
    color: var(--color-warm-cream);
    padding: 6px 12px;
    background: var(--color-graphite);
    border: 1px solid var(--color-steel);
    border-radius: var(--radius-pill);
  }}

  .agy-nav-btn {{
    background: var(--color-carbon);
    color: var(--color-porcelain);
    border: 1px solid var(--color-steel);
    padding: 8px 14px;
    border-radius: var(--radius-pill);
    font-family: var(--font-sans);
    font-size: 12px;
    line-height: 16px;
    font-weight: 600;
    letter-spacing: 0.025em;
    text-transform: uppercase;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: transform 150ms var(--ease-out),
                background-color 150ms var(--ease-out),
                border-color 150ms var(--ease-out);
  }}
  .agy-nav-btn.primary-iris {{
    background: var(--color-electric-iris);
    color: var(--color-bone-white);
    border-color: transparent;
  }}
  .agy-nav-btn:active:not(:disabled) {{
    transform: scale(0.97);
  }}
  .agy-nav-btn:disabled {{
    opacity: 0.32;
    cursor: not-allowed;
  }}

  /* ═══════════════════════════════════════════════════════════════
     CONTENEDOR PRINCIPAL DE LA EXPERIENCIA (8 DIAPOSITIVAS)
     ═══════════════════════════════════════════════════════════════ */
  #agy-app-shell {{
    position: fixed;
    top: 60px;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 9500;
    background: var(--color-void);
    color: var(--color-porcelain);
    font-family: var(--font-sans);
    display: flex;
    flex-direction: column;
    opacity: 0;
    pointer-events: none;
    transform: translateY(4px) scale(0.996);
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
  #agy-app-shell.keyboard-instant {{
    transition-duration: 0ms !important;
  }}

  /* ═══════════════════════════════════════════════════════════════
     ESCENARIO EDITORIAL (DIAPOSITIVAS 1, 6, 7 Y 8)
     Apple Midnight Hardware Gallery × ORYZO Darkroom × Dala Void
     ═══════════════════════════════════════════════════════════════ */
  #agy-editorial-stage {{
    flex: 1;
    width: 100%;
    height: 100%;
    overflow-y: auto;
    background: var(--color-void);
    position: relative;
    display: none;
  }}
  #agy-editorial-stage.active {{
    display: block;
  }}

  .agy-ed-slide {{
    display: none;
    max-width: 1320px;
    margin: 0 auto;
    padding: 36px 36px 72px 36px;
    position: relative;
    z-index: 2;
  }}
  .agy-ed-slide.active {{
    display: block;
    animation: agySlideEnter 260ms var(--ease-out) both;
  }}
  @keyframes agySlideEnter {{
    from {{
      opacity: 0;
      transform: translateY(8px);
    }}
    to {{
      opacity: 1;
      transform: translateY(0);
    }}
  }}

  /* Hero de Dos Columnas Dala + Constelación en Vivo (Diapositiva 1) */
  .agy-hero-grid {{
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 36px;
    align-items: center;
    padding-bottom: 36px;
    border-bottom: 1px dashed var(--color-cork-border);
    margin-bottom: 36px;
  }}
  .agy-hero-title {{
    font-size: 46px;
    font-weight: 500;
    line-height: 1.08;
    letter-spacing: -0.026em;
    color: var(--color-bone-white);
    margin: 0 0 18px 0;
  }}
  .agy-hero-title em {{
    font-family: var(--font-serif);
    font-style: italic;
    font-weight: 400;
    color: var(--color-warm-cream);
  }}
  .agy-hero-lead {{
    font-size: 17px;
    font-weight: 400;
    line-height: 1.62;
    color: var(--color-silver-mist);
    max-width: 64ch;
    margin-bottom: 24px;
  }}
  .agy-meta-authors {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px 24px;
    padding: 16px 0;
    border-top: 1px dashed var(--color-cork-border);
    border-bottom: 1px dashed var(--color-cork-border);
    margin-bottom: 24px;
    font-size: 14px;
    color: var(--color-warm-cream);
  }}
  .agy-meta-authors span {{
    color: var(--color-ash-gray);
  }}
  .agy-cta-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 14px;
  }}
  .agy-pill-cta {{
    background: var(--color-electric-iris);
    color: var(--color-bone-white);
    border: none;
    border-radius: 24px;
    padding: 14px 24px;
    font-family: var(--font-sans);
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.025em;
    text-transform: uppercase;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    transition: transform 160ms var(--ease-out), background-color 160ms var(--ease-out);
  }}
  .agy-pill-cta:active {{
    transform: scale(0.97);
  }}
  .agy-pill-ghost {{
    background: transparent;
    color: var(--color-warm-cream);
    border: 1px solid var(--color-cork-border);
    border-radius: 24px;
    padding: 13px 20px;
    font-family: var(--font-sans);
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    cursor: pointer;
    transition: transform 160ms var(--ease-out), border-color 160ms var(--ease-out), color 160ms var(--ease-out);
  }}
  .agy-pill-ghost:active {{
    transform: scale(0.97);
  }}

  /* Plinto de Constelación Interactiva Dala en la Portada */
  .agy-constellation-plinth {{
    position: relative;
    height: 390px;
    background: var(--color-obsidian);
    border: 1px solid var(--color-steel);
    border-radius: var(--radius-cards);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 24px;
  }}
  #agy-hero-constellation {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
    cursor: crosshair;
  }}
  .agy-plinth-caption {{
    position: relative;
    z-index: 2;
    background: rgba(14, 14, 14, 0.82);
    backdrop-filter: blur(12px);
    border: 1px solid var(--color-steel);
    border-radius: 18px;
    padding: 14px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    pointer-events: none;
  }}

  /* Grillas y Tarjetas Apple Midnight (28px radius, 1px Steel border, 0 drop-shadow) */
  .agy-bento-grid-4 {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
    margin-bottom: 30px;
  }}
  .agy-bento-grid-2 {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 24px;
    margin-bottom: 30px;
  }}
  .agy-bento-grid-3 {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-bottom: 30px;
  }}

  .agy-plinth-card {{
    background: linear-gradient(180deg, #1d1d1f 0%, #0a0a0c 100%);
    border: 1px solid var(--color-steel);
    border-radius: var(--radius-cards);
    padding: 24px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 12px;
  }}
  .agy-plinth-card.obsidian {{
    background: var(--color-obsidian);
  }}
  .agy-plinth-card.highlight-iris {{
    border-color: rgba(128, 82, 255, 0.55);
  }}
  .agy-plinth-card.highlight-saffron {{
    border-color: rgba(255, 184, 41, 0.55);
  }}

  .agy-stat-num {{
    font-family: var(--font-sans);
    font-size: 38px;
    font-weight: 500;
    line-height: 1.05;
    letter-spacing: -0.03em;
    color: var(--color-porcelain);
  }}
  .agy-stat-unit {{
    font-size: 16px;
    font-weight: 400;
    color: var(--color-saffron-spark);
    margin-left: 4px;
  }}
  .agy-stat-title {{
    font-size: 15px;
    font-weight: 600;
    line-height: 1.35;
    color: var(--color-warm-cream);
  }}
  .agy-stat-desc {{
    font-size: 14px;
    font-weight: 400;
    line-height: 1.55;
    color: var(--color-silver-mist);
  }}

  /* Banner Matemático Monospaced Original (Dala / Terminal Style) */
  .agy-formula-banner {{
    background: #0d0e12;
    border: 1px dashed rgba(255, 184, 41, 0.4);
    border-radius: 12px;
    padding: 14px 20px;
    font-family: var(--font-mono);
    font-size: 13.5px;
    line-height: 1.6;
    color: var(--color-warm-cream);
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 24px;
  }}
  .agy-formula-banner .agy-formula-title {{
    color: var(--color-saffron-spark);
    font-weight: 700;
    letter-spacing: 0.02em;
  }}
  .agy-formula-banner .agy-formula-eq {{
    color: var(--color-porcelain);
    margin-left: 8px;
  }}
  .agy-formula-banner .agy-formula-sub {{
    color: var(--color-ash-gray);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }}

  /* Módulo Matemático Editorial (Reemplaza el bloque monoespaciado crudo) */
  .agy-math-showcase {{
    background: linear-gradient(180deg, #161618 0%, #0a0a0c 100%);
    border: 1px solid var(--color-steel);
    border-radius: var(--radius-cards);
    padding: 24px 28px;
    margin-bottom: 28px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }}
  .agy-math-top {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding-bottom: 18px;
    border-bottom: 1px dashed var(--color-cork-border);
  }}
  .agy-math-heading {{
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .agy-math-heading h2,
  .agy-math-heading h3 {{
    font-size: 20px;
    font-weight: 600;
    line-height: 1.3;
    color: var(--color-warm-cream);
    letter-spacing: -0.01em;
    margin: 0;
  }}
  .agy-math-heading p {{
    font-size: 14px;
    line-height: 1.45;
    color: var(--color-ash-gray);
  }}
  .agy-math-eq {{
    background: var(--color-obsidian);
    border: 1px solid rgba(255, 184, 41, 0.35);
    border-radius: 16px;
    padding: 12px 22px;
    font-family: var(--font-serif);
    font-size: 21px;
    line-height: 1.35;
    color: var(--color-porcelain);
    display: inline-block;
    letter-spacing: 0.01em;
    white-space: nowrap;
  }}
  .agy-math-eq var {{
    font-family: var(--font-serif);
    font-style: italic;
    font-weight: 500;
    color: var(--color-warm-cream);
  }}
  .agy-math-eq sub {{
    font-family: var(--font-sans);
    font-style: normal;
    font-size: 12px;
    font-weight: 600;
    color: var(--color-saffron-spark);
    margin-left: 1px;
    margin-right: 4px;
  }}
  .agy-math-params {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
  }}
  .agy-math-param-card {{
    background: var(--color-obsidian);
    border: 1px solid var(--color-steel);
    border-radius: 18px;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .agy-math-param-val {{
    font-family: var(--font-sans);
    font-size: 18px;
    font-weight: 600;
    line-height: 1.3;
    color: var(--color-saffron-spark);
    font-variant-numeric: tabular-nums;
  }}
  .agy-math-param-lbl {{
    font-size: 13px;
    font-weight: 600;
    line-height: 1.35;
    color: var(--color-porcelain);
  }}
  .agy-math-param-sub {{
    font-size: 12px;
    font-weight: 400;
    line-height: 1.4;
    color: var(--color-ash-gray);
  }}

  /* Contenedor del Mapa Folium Original en Diapositiva 2 (Modo Autónomo) */
  #agy-folium-stage {{
    flex: 1;
    width: 100%;
    height: 100%;
    display: none;
    background: var(--color-void);
    position: relative;
  }}
  #agy-folium-stage.active {{
    display: block;
  }}
  #agy-folium-iframe {{
    width: 100%;
    height: 100%;
    border: none;
    display: block;
  }}

  /* ═══════════════════════════════════════════════════════════════
     ESCENARIO INTERACTIVO DE GRAFOS Y MAPA SIG (DIAPOSITIVAS 2, 3, 4 Y 5)
     ═══════════════════════════════════════════════════════════════ */
  #agy-interactive-stage {{
    flex: 1;
    display: none;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
  }}
  #agy-interactive-stage.active {{
    display: flex;
  }}

  .agy-deck-header {{
    min-height: 58px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 24px;
    background: var(--color-graphite);
    border-bottom: 1px solid var(--color-steel);
    flex-wrap: wrap;
    gap: 12px;
  }}
  .agy-deck-title-box {{
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .agy-deck-title-box h2 {{
    font-size: 20px;
    line-height: 26px;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: var(--color-porcelain);
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .agy-section-tag {{
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
    padding: 3px 10px;
    border-radius: var(--radius-pill);
    background: rgba(128, 82, 255, 0.16);
    color: var(--color-saffron-spark);
    border: 1px solid rgba(128, 82, 255, 0.45);
    font-weight: 600;
    text-transform: uppercase;
  }}
  .agy-deck-title-box p {{
    font-size: 14px;
    line-height: 20px;
    color: var(--color-ash-gray);
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
    background: var(--color-void);
    padding: 3px;
    border-radius: var(--radius-pill);
    border: 1px solid var(--color-steel);
    gap: 2px;
  }}
  .agy-ctrl-btn, .agy-ctrl-select {{
    background: var(--color-carbon);
    color: var(--color-porcelain);
    border: 1px solid var(--color-steel);
    padding: 7px 14px;
    border-radius: var(--radius-pill);
    font-family: var(--font-sans);
    font-size: 12px;
    line-height: 16px;
    font-weight: 500;
    cursor: pointer;
    transition: transform 150ms var(--ease-out),
                border-color 150ms var(--ease-out),
                background-color 150ms var(--ease-out),
                color 150ms var(--ease-out);
  }}
  .agy-ctrl-group .agy-ctrl-btn {{
    border: none;
    background: transparent;
    color: var(--color-ash-gray);
    padding: 5px 12px;
  }}
  .agy-ctrl-btn:active {{
    transform: scale(0.97);
  }}
  .agy-ctrl-btn.active, .agy-ctrl-group .agy-ctrl-btn.active {{
    background: var(--color-electric-iris);
    color: var(--color-bone-white);
    font-weight: 600;
  }}

  /* Cuerpo del Explorador: Lienzo Constellation #000000 + Inspector Técnico de 432px */
  .agy-deck-body {{
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 432px;
    min-height: 0;
    overflow: hidden;
  }}

  .agy-canvas-wrap {{
    position: relative;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background-color: var(--color-void);
  }}

  /* Lienzo de Partículas Triangulares Dala detrás del Grafo SVG */
  #agy-constellation-canvas {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
  }}

  #agy-graph-svg {{
    position: relative;
    z-index: 5;
    width: 100%;
    height: 100%;
    cursor: grab;
    display: block;
  }}
  #agy-graph-svg:active {{
    cursor: grabbing;
  }}

  /* Tarjetas KPI Flotantes sobre el Lienzo (Apple Carbon Capsules) */
  .agy-kpi-strip {{
    position: absolute;
    top: 14px;
    left: 16px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    pointer-events: none;
    z-index: 20;
  }}
  .agy-kpi-card {{
    background: rgba(14, 14, 14, 0.88);
    border: 1px solid var(--color-steel);
    border-radius: 16px;
    padding: 8px 14px;
    backdrop-filter: blur(12px);
  }}
  .agy-kpi-val {{
    font-family: var(--font-mono);
    font-size: 16px;
    line-height: 22px;
    font-weight: 600;
    color: var(--color-warm-cream);
  }}
  .agy-kpi-lbl {{
    font-size: 12px;
    line-height: 16px;
    color: var(--color-ash-gray);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}

  /* Panel Flotante de Capas SIG y Criterios (Diapositiva 2: Mapa Geográfico) */
  #agy-gis-layers-panel {{
    position: absolute;
    top: 82px;
    left: 16px;
    width: 264px;
    background: rgba(14, 14, 14, 0.92);
    border: 1px solid var(--color-steel);
    border-radius: 20px;
    padding: 14px;
    backdrop-filter: blur(12px);
    z-index: 25;
    display: none;
  }}
  #agy-gis-layers-panel.visible {{
    display: block;
  }}
  .agy-gis-panel-title {{
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-warm-cream);
    margin: 0 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px dashed var(--color-cork-border);
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
    font-size: 13px;
    line-height: 18px;
    color: var(--color-porcelain);
    cursor: pointer;
  }}
  .agy-layer-toggle input {{
    accent-color: var(--color-electric-iris);
    cursor: pointer;
  }}
  .agy-layer-count {{
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--color-ash-gray);
  }}
  .agy-cam-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-top: 8px;
  }}
  .agy-cam-btn {{
    background: var(--color-carbon);
    color: var(--color-silver-mist);
    border: 1px solid var(--color-steel);
    border-radius: 10px;
    padding: 6px 8px;
    font-size: 12px;
    line-height: 16px;
    cursor: pointer;
    text-align: left;
    transition: transform 150ms var(--ease-out),
                color 150ms var(--ease-out),
                border-color 150ms var(--ease-out);
  }}
  .agy-cam-btn:active {{
    transform: scale(0.97);
  }}

  /* HUD Cartográfico Superior Derecha */
  #agy-coords-hud {{
    position: absolute;
    top: 14px;
    right: 16px;
    background: rgba(14, 14, 14, 0.88);
    border: 1px solid var(--color-steel);
    border-radius: var(--radius-pill);
    padding: 6px 14px;
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
    color: var(--color-warm-cream);
    display: flex;
    align-items: center;
    gap: 12px;
    pointer-events: none;
    z-index: 20;
  }}

  /* Barra Inferior de Filtro por Rol Funcional de Vértice */
  .agy-type-filter-bar {{
    position: absolute;
    bottom: 14px;
    left: 16px;
    background: rgba(14, 14, 14, 0.92);
    border: 1px solid var(--color-steel);
    border-radius: var(--radius-pill);
    padding: 6px 12px;
    font-size: 12px;
    line-height: 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    max-width: calc(100% - 32px);
    backdrop-filter: blur(12px);
    z-index: 20;
  }}
  .agy-filter-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--color-silver-mist);
    background: transparent;
    border: 1px solid transparent;
    padding: 4px 10px;
    border-radius: var(--radius-pill);
    font-size: 12px;
    line-height: 16px;
    cursor: pointer;
    transition: transform 150ms var(--ease-out),
                background-color 150ms var(--ease-out),
                border-color 150ms var(--ease-out),
                color 150ms var(--ease-out);
  }}
  .agy-filter-pill:active {{
    transform: scale(0.97);
  }}
  .agy-filter-pill.active {{
    background: var(--color-electric-iris);
    color: var(--color-bone-white);
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

  /* Inspector Lateral Derecho: Dala Void + Apple Carbon Cards */
  .agy-inspector {{
    background: var(--color-graphite);
    border-left: 1px solid var(--color-steel);
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
    position: relative;
    z-index: 10;
  }}
  .agy-inspector-header {{
    padding: 14px 18px;
    background: var(--color-obsidian);
    border-bottom: 1px dashed var(--color-cork-border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }}
  .agy-inspector-header h3 {{
    margin: 0;
    font-size: 15px;
    line-height: 20px;
    font-weight: 600;
    color: var(--color-warm-cream);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .agy-inspector-content {{
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    font-size: 13px;
    line-height: 19px;
  }}

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
    background: var(--color-carbon);
    border: 1px solid var(--color-steel);
    border-radius: 20px;
    padding: 14px 16px;
    margin-bottom: 14px;
    animation: agyStaggerIn 220ms var(--ease-out) both;
  }}
  .agy-prop-group:nth-child(1) {{ animation-delay: 0ms; }}
  .agy-prop-group:nth-child(2) {{ animation-delay: 35ms; }}
  .agy-prop-group:nth-child(3) {{ animation-delay: 65ms; }}
  .agy-prop-group:nth-child(4) {{ animation-delay: 90ms; }}

  .agy-prop-group h4 {{
    margin: 0 0 10px 0;
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-warm-cream);
    border-bottom: 1px dashed var(--color-cork-border);
    padding-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .agy-prop-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 5px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    gap: 8px;
  }}
  .agy-prop-row:last-child {{
    border-bottom: none;
  }}
  .agy-prop-key {{
    color: var(--color-ash-gray);
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
  }}
  .agy-prop-val {{
    color: var(--color-porcelain);
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
    font-weight: 600;
    text-align: right;
  }}

  .agy-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: var(--radius-pill);
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
    font-weight: 600;
    border: 1px solid var(--color-steel);
    flex-shrink: 0;
  }}

  .agy-effect-box {{
    background: rgba(128, 82, 255, 0.10);
    border: 1px solid rgba(128, 82, 255, 0.38);
    padding: 12px 14px;
    border-radius: 18px;
    color: var(--color-porcelain);
    font-size: 13px;
    line-height: 19px;
    margin-bottom: 14px;
    animation: agyStaggerIn 180ms var(--ease-out) both;
  }}

  .agy-code-block {{
    background: var(--color-void);
    border: 1px solid var(--color-steel);
    border-radius: 14px;
    padding: 12px;
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 17px;
    color: var(--color-warm-cream);
    overflow-x: auto;
    white-space: pre;
    max-height: 220px;
  }}

  .agy-chip-list {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
  }}
  .agy-chip {{
    background: var(--color-void);
    border: 1px solid var(--color-steel);
    color: var(--color-porcelain);
    padding: 5px 10px;
    border-radius: var(--radius-pill);
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: transform 150ms var(--ease-out),
                border-color 150ms var(--ease-out),
                background-color 150ms var(--ease-out),
                color 150ms var(--ease-out);
  }}
  .agy-chip:active {{
    transform: scale(0.96);
  }}

  /* Tablas técnicas de ingeniería */
  .agy-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 13px;
    line-height: 18px;
  }}
  .agy-table th {{
    text-align: left;
    color: var(--color-warm-cream);
    padding: 8px 10px;
    border-bottom: 1px dashed var(--color-cork-border);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 12px;
  }}
  .agy-table td {{
    padding: 8px 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    color: var(--color-porcelain);
  }}
  .agy-table tr.row-optimal td {{
    background: rgba(128, 82, 255, 0.14);
    color: var(--color-saffron-spark);
    font-weight: 600;
  }}

  /* Barras de progreso de composición GPU (transform: scaleX) */
  @keyframes agyScaleBar {{
    from {{ transform: scaleX(0); }}
    to {{ transform: scaleX(1); }}
  }}
  .agy-bar-wrap {{
    margin: 8px 0;
  }}
  .agy-bar-label {{
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
    color: var(--color-silver-mist);
  }}
  .agy-bar-bg {{
    height: 8px;
    background: var(--color-void);
    border: 1px solid var(--color-steel);
    border-radius: 9999px;
    overflow: hidden;
    margin-top: 4px;
  }}
  .agy-bar-fill {{
    height: 100%;
    background: var(--color-electric-iris);
    border-radius: 9999px;
    transform-origin: left center;
    animation: agyScaleBar 260ms var(--ease-out) both;
  }}

  /* Elementos SVG del Grafo Constellation y del Mapa SIG */
  .agy-edge-path {{
    fill: none;
    cursor: pointer;
    transition: stroke-width 150ms var(--ease-out),
                stroke-opacity 150ms var(--ease-out);
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
    transform: scale(0.94);
  }}

  @keyframes agyDashFlow {{
    to {{ stroke-dashoffset: -36; }}
  }}
  .agy-route-highlight {{
    animation: agyDashFlow 1.1s linear infinite;
  }}

  /* Hover Gating & Prefers-Reduced-Motion (STANDARDS.md) */
  @media (hover: hover) and (pointer: fine) {{
    .agy-nav-btn:hover:not(:disabled) {{
      border-color: var(--color-electric-iris);
      color: var(--color-bone-white);
    }}
    .agy-pill-cta:hover {{
      background: #5827e0;
    }}
    .agy-pill-ghost:hover {{
      border-color: var(--color-warm-cream);
      color: var(--color-bone-white);
    }}
    .agy-slide-tab:hover:not(.active) {{
      color: var(--color-bone-white);
      background: rgba(255, 255, 255, 0.05);
    }}
    .agy-ctrl-btn:hover:not(.active), .agy-ctrl-select:hover, .agy-cam-btn:hover {{
      border-color: var(--color-electric-iris);
      color: var(--color-bone-white);
    }}
    .agy-chip:hover {{
      border-color: var(--color-electric-iris);
      background: rgba(128, 82, 255, 0.16);
      color: var(--color-bone-white);
    }}
    .agy-filter-pill:hover:not(.active) {{
      background: var(--color-carbon);
      color: var(--color-bone-white);
    }}
    .agy-edge-path:hover {{
      stroke-width: 4.2px !important;
      stroke-opacity: 1 !important;
    }}
    .agy-node-group:hover .agy-node-circle {{
      transform: scale(1.16);
      stroke-width: 3.2px;
    }}
  }}

  @media (prefers-reduced-motion: reduce) {{
    #agy-app-shell, .agy-ed-slide, .agy-prop-group, .agy-effect-box, .agy-bar-fill,
    .agy-nav-btn, .agy-slide-tab, .agy-ctrl-btn, .agy-chip, .agy-node-circle {{
      animation: none !important;
      transition: opacity 150ms ease, background-color 150ms ease, border-color 150ms ease !important;
      transform: none !important;
    }}
    .agy-route-highlight {{
      animation: none !important;
    }}
  }}

  @media (max-width: 1100px) {{
    .agy-hero-grid, .agy-bento-grid-4, .agy-bento-grid-3, .agy-bento-grid-2 {{
      grid-template-columns: 1fr;
    }}
    .agy-brand-sub, #agy-oryzo-serial {{
      display: none;
    }}
    .agy-deck-body {{
      grid-template-columns: 1fr;
      grid-template-rows: 1fr 310px;
    }}
    .agy-inspector {{
      border-left: none;
      border-top: 1px solid var(--color-steel);
    }}
  }}
</style>

<!-- Etiqueta Serial Vertical estilo ORYZO AI -->
<div id="agy-oryzo-serial" aria-hidden="true">UNILLANOS · ING. SISTEMAS · 603301 · PUERTO GAITÁN, META · G=(V,E)</div>

<!-- ═══════════════════════════════════════════════════════════════════════
     BARRA GLOBAL DE NAVEGACIÓN (8 DIAPOSITIVAS DE LA EXPOSICIÓN INTERACTIVA)
     ═══════════════════════════════════════════════════════════════════════ -->
<header id="agy-slide-nav" role="navigation" aria-label="Navegación de presentación logística">
  <div class="agy-brand">
    <div class="agy-brand-mark" aria-hidden="true">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="12 3 21 19 3 19"/>
        <circle cx="12" cy="14" r="2.2" fill="#ffb829" stroke="none"/>
      </svg>
    </div>
    <div class="agy-brand-text">
      <span class="agy-brand-title">UNILLANOS · Red Logística G=(V,E)</span>
      <span class="agy-brand-sub">Puerto Gaitán, Meta · 25 Vértices · 41 Aristas · FLP</span>
    </div>
  </div>

  <div class="agy-slide-tabs" role="tablist" aria-label="Diapositivas">
    <button class="agy-slide-tab active is-editorial" role="tab" aria-selected="true" data-slide="0" onclick="agyCambiarSlide(0, false)">
      <span class="agy-tab-dot"></span>1. Portada
    </button>
    <button class="agy-slide-tab" role="tab" aria-selected="false" data-slide="1" onclick="agyCambiarSlide(1, false)">
      <span class="agy-tab-dot"></span>2. Mapa SIG
    </button>
    <button class="agy-slide-tab" role="tab" aria-selected="false" data-slide="2" onclick="agyCambiarSlide(2, false)">
      <span class="agy-tab-dot"></span>3. Grafo Vial
    </button>
    <button class="agy-slide-tab" role="tab" aria-selected="false" data-slide="3" onclick="agyCambiarSlide(3, false)">
      <span class="agy-tab-dot"></span>4. Grafo Ríos
    </button>
    <button class="agy-slide-tab" role="tab" aria-selected="false" data-slide="4" onclick="agyCambiarSlide(4, false)">
      <span class="agy-tab-dot"></span>5. Grafo Mixto
    </button>
    <button class="agy-slide-tab is-editorial" role="tab" aria-selected="false" data-slide="5" onclick="agyCambiarSlide(5, false)">
      <span class="agy-tab-dot"></span>6. Algoritmos
    </button>
    <button class="agy-slide-tab is-editorial" role="tab" aria-selected="false" data-slide="6" onclick="agyCambiarSlide(6, false)">
      <span class="agy-tab-dot"></span>7. Solución FLP
    </button>
    <button class="agy-slide-tab is-editorial" role="tab" aria-selected="false" data-slide="7" onclick="agyCambiarSlide(7, false)">
      <span class="agy-tab-dot"></span>8. Impacto
    </button>
  </div>

  <div class="agy-nav-actions">
    <span id="agy-slide-counter" aria-live="polite">Vista 1 / 8</span>
    <button id="agy-btn-prev" class="agy-nav-btn" onclick="agyCambiarSlide(agySlideActual - 1, false)" disabled aria-label="Diapositiva anterior">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><polyline points="15 18 9 12 15 6"/></svg>
      Anterior
    </button>
    <button id="agy-btn-next" class="agy-nav-btn primary-iris" onclick="agyCambiarSlide(agySlideActual + 1, false)" aria-label="Siguiente diapositiva">
      Siguiente
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><polyline points="9 18 15 12 9 6"/></svg>
    </button>
  </div>
</header>

<!-- ═══════════════════════════════════════════════════════════════════════
     SHELL PRINCIPAL (ALTERNA ENTRE EXHIBICIÓN EDITORIAL Y EXPLORADOR G(V,E))
     ═══════════════════════════════════════════════════════════════════════ -->
<main id="agy-app-shell" class="{clase_slide0}">

  <!-- ─────────────────────────────────────────────────────────────────────
       BLOQUE A: ESCENARIO EDITORIAL INTERACTIVO (DIAPOSITIVAS 1, 6, 7 Y 8)
       ───────────────────────────────────────────────────────────────────── -->
  <section id="agy-editorial-stage" class="active" aria-label="Presentación Ejecutiva del Proyecto">

    <!-- DIAPOSITIVA 1: PORTADA, CONTEXTO TERRITORIAL, PROBLEMA Y CINÉTICA TÉRMICA -->
    <article id="agy-ed-slide-0" class="agy-ed-slide active">
      <div class="agy-hero-grid">
        <div>
          <h1 class="agy-hero-title">
            Optimización Multimodal de <em>Cadenas de Frío</em> para Pesca Artesanal mediante Teoría de Grafos y FLP.
          </h1>
          <p class="agy-hero-lead">
            Modelado matemático sobre grafo dirigido ponderado G=(V,E) para erradicar la pérdida poscaptura de cachama (Colossoma macropomum) en las cuencas de los ríos Meta, Manacacías e Iteviare — Corredor Puerto Gaitán, Meta hacia Villavicencio y Bogotá.
          </p>
          <div class="agy-meta-authors">
            <div><strong>UNIVERSIDAD DE LOS LLANOS</strong> <span>· Fac. Ciencias Básicas e Ingeniería · Ing. en Sistemas</span></div>
            <div><strong>Autores:</strong> Yair Armando Ospino Herrera (160005565) · Harold Esteban Padilla Díaz (160005567)</div>
            <div><strong>Curso:</strong> Estructuras de Datos (603301) · Corte 3: Grafos · Semestre 2026-II · Villavicencio, Meta</div>
          </div>
          <div class="agy-cta-row">
            <button class="agy-pill-cta" onclick="agyCambiarSlide(4, false)">
              Explorar Constelación Multimodal G=(V,E)
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
            </button>
            <button class="agy-pill-ghost" onclick="agyCambiarSlide(1, false)">Ver Mapa Geográfico SIG</button>
            <button class="agy-pill-ghost" onclick="agyCambiarSlide(6, false)">Ver Solución Óptima FLP</button>
          </div>
        </div>

        <!-- Constelación Interactiva en Vivo Dala (Partículas Triangulares + Nodos) -->
        <div class="agy-constellation-plinth">
          <canvas id="agy-hero-constellation" aria-label="Visualización de constelación de nodos logísticos"></canvas>
          <div class="agy-plinth-caption">
            <div>
              <div style="font-size:13px;font-weight:600;color:var(--color-warm-cream);">Constelación Topológica · Cuenca del Río Meta</div>
              <div style="font-family:var(--font-mono);font-size:12px;color:var(--color-ash-gray);">25 Vértices · 41 Aristas · 12,413 Vértices Fluviales IGAC</div>
            </div>
            <span class="agy-badge" style="color:var(--color-saffron-spark);border-color:rgba(255,184,41,0.4);">INTERACTIVO</span>
          </div>
        </div>
      </div>

      <!-- El Problema Cuantificado (4 Plintos Apple Carbon 28px) -->
      <div class="agy-bento-grid-4">
        <div class="agy-plinth-card highlight-saffron">
          <div class="agy-stat-num">265.8<span class="agy-stat-unit">kg/día</span></div>
          <div class="agy-stat-title">Producción Censada en 7 Asociaciones</div>
          <div class="agy-stat-desc">ASOPESGA, Atarraya Verde, Asovida, Asopespor, Puerto Trujillo, Desembarcadero y Soplaviento.</div>
        </div>
        <div class="agy-plinth-card">
          <div class="agy-stat-num" style="color:var(--color-ember-accent);">40.0<span class="agy-stat-unit">%</span></div>
          <div class="agy-stat-title">Pérdida Poscaptura Sin Cadena de Frío</div>
          <div class="agy-stat-desc">Exposición a 30 °C de temperatura ambiente y radiación solar directa sobre canoas sin hielo.</div>
        </div>
        <div class="agy-plinth-card highlight-iris">
          <div class="agy-stat-num">4.0<span class="agy-stat-unit">h (240 min)</span></div>
          <div class="agy-stat-title">Límite Estricto de Inocuidad FAO</div>
          <div class="agy-stat-desc">Umbral sanitario máximo admisible antes de que el pescado fresco pierda aptitud comercial.</div>
        </div>
        <div class="agy-plinth-card">
          <div class="agy-stat-num">63.0<span class="agy-stat-unit">km</span></div>
          <div class="agy-stat-title">Aislamiento Fluvial Crítico (P05)</div>
          <div class="agy-stat-desc">Comunidades como Puerto Trujillo en el río Iteviare operan sin energía eléctrica continua ni muelles fríos.</div>
        </div>
      </div>

      <!-- Fundamento Biológico: Cinética Térmica de la Cachama (Banner Original Solicitado) -->
      <div class="agy-formula-banner">
        <div>
          <span class="agy-formula-title">MODELO MATEMÁTICO TÉRMICO (Cachama · Colossoma macropomum):</span>
          <span class="agy-formula-eq">D_acum = D_0 + &Sigma; (k_i &middot; t_i)</span>
        </div>
        <div class="agy-formula-sub">
          k_base = 2.5%/h (30 °C) &middot; Zona D = &times;1.5 (3.75%/h) &middot; Muelle M = +0.25 h al sol &middot; Centro Frío F = 0 °C (reinicia t = 0.0 h, vida útil 360 h)
        </div>
      </div>

      <!-- Fundamento Biológico: Cinética Térmica de la Cachama (Módulo Matemático Refinado) -->
      <div class="agy-math-showcase">
        <div class="agy-math-top">
          <div class="agy-math-heading">
            <h2>Cinética de Degradación Térmica · Cachama de Río (<em style="font-family:var(--font-serif);font-style:italic;font-weight:400;">Colossoma macropomum</em>)</h2>
            <p>Función acumulativa de pérdida poscaptura por tramos antes del ingreso a la cadena de frío.</p>
          </div>
          <div class="agy-math-eq" aria-label="D acumulada igual a D sub cero más sumatoria de k sub i por t sub i">
            <var>D</var><sub>acum</sub> <span>=</span> <var>D</var><sub>0</sub> <span>+</span> <span>&Sigma;</span> ( <var>k</var><sub>i</sub> &middot; <var>t</var><sub>i</sub> )
          </div>
        </div>
        <div class="agy-math-params">
          <div class="agy-math-param-card">
            <span class="agy-math-param-val">2.5% / hora</span>
            <span class="agy-math-param-lbl">Tasa Base (<var style="font-family:var(--font-serif);font-style:italic;">k</var><sub>base</sub> a 30 °C)</span>
            <span class="agy-math-param-sub">Degradación estándar en canoa artesanal sin hielo</span>
          </div>
          <div class="agy-math-param-card">
            <span class="agy-math-param-val" style="color:var(--color-ember-accent);">×1.5 (3.75% / h)</span>
            <span class="agy-math-param-lbl">Zonas Críticas de Deterioro (D)</span>
            <span class="agy-math-param-sub">Multiplicador térmico en pasos descubiertos sin sombra</span>
          </div>
          <div class="agy-math-param-card">
            <span class="agy-math-param-val" style="color:var(--color-electric-link);">+0.25 h (+15 min)</span>
            <span class="agy-math-param-lbl">Penalización de Muelle (M)</span>
            <span class="agy-math-param-sub">Espera bajo radiación solar en transbordo lancha → camión</span>
          </div>
          <div class="agy-math-param-card">
            <span class="agy-math-param-val" style="color:#10b981;">0 °C &rarr; t = 0.0 h</span>
            <span class="agy-math-param-lbl">Abatimiento en Centro Frío (F)</span>
            <span class="agy-math-param-sub">Reinicia el reloj térmico y extiende la vida útil a 360 h (15 días)</span>
          </div>
        </div>
      </div>
    </article>

    <!-- DIAPOSITIVA 6: NÚCLEO ALGORÍTMICO, MODELO MATEMÁTICO Y DEFENSA TÉCNICA -->
    <article id="agy-ed-slide-5" class="agy-ed-slide">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:16px;padding-bottom:24px;border-bottom:1px dashed var(--color-cork-border);margin-bottom:28px;">
        <div>
          <h2>Proceso Algorítmico, Modelo Matemático y Criterios de Decisión</h2>
          <p style="margin-top:6px;">Implementación 100% propia desde cero sobre Lista de Adyacencia Hash O(1) — sin librerías externas de cálculo.</p>
        </div>
        <div class="agy-cta-row">
          <button class="agy-pill-ghost" onclick="agyCambiarSlide(4, false)">Ver Grafo en Vivo</button>
          <button class="agy-pill-cta" onclick="agyCambiarSlide(6, false)">Ver Validación FLP →</button>
        </div>
      </div>

      <!-- Simulador Interactivo de la Función Objetivo Multicriterio (ADR-02) -->
      <div class="agy-math-showcase">
        <div class="agy-math-top" style="border-bottom:none;padding-bottom:0;">
          <div class="agy-math-heading">
            <h3>Función Objetivo de Peso Multicriterio por Arista (ADR-02)</h3>
            <p>Pondera 6 dimensiones operativas normalizadas para hallar rutas óptimas resilientes.</p>
          </div>
          <div class="agy-math-eq">
            <var>w</var><sub>ij</sub> <span>=</span> 0.25<var>t&#770;</var> <span>+</span> 0.20<var>c&#770;</var> <span>+</span> 0.20<var>r</var><sub>comb</sub> <span>+</span> 0.10<var>p&#770;</var> <span>+</span> 0.10(1&minus;<var>q</var>) <span>+</span> 0.05<var>P</var><sub>ret</sub>
          </div>
          <div style="display:flex;align-items:center;gap:10px;">
            <label for="agy-ed-edge-sim" style="font-size:13px;font-weight:500;color:var(--color-warm-cream);">Auditar Arista:</label>
            <select id="agy-ed-edge-sim" class="agy-ctrl-select" onchange="agyActualizarSimuladorMulticriterio(this.value)"></select>
          </div>
        </div>
      </div>

      <div class="agy-bento-grid-2">
        <div class="agy-plinth-card highlight-iris" id="agy-ed-multicriterio-box"></div>
        <div class="agy-plinth-card">
          <h3>Verificación Cruzada: Bellman-Ford O(V·E) vs. Dijkstra O((V+E) log V)</h3>
          <p style="font-size:14px;">Coincidencia exacta del 100% en tiempos mínimos desde P01 con filtro ambiental activo (evitar_restringidos=True) y detección formal de ciclos negativos.</p>
          <div id="agy-ed-bf-table-wrap" style="overflow-x:auto;margin-top:8px;"></div>
        </div>
      </div>

      <!-- Los 4 Pilares Algorítmicos y Recorridos Canónicos BFS / DFS -->
      <div class="agy-bento-grid-2">
        <div class="agy-plinth-card obsidian">
          <h3>Recorridos Canónicos del Curso: BFS O(V+E) y DFS (Backtracking)</h3>
          <p style="font-size:14px;">BFS garantiza el menor número de saltos/transbordos hacia el centro frigorífico F02; DFS enumera todas las rutas simples para resiliencia ante contingencias.</p>
          <div id="agy-ed-bfs-dfs-wrap" style="overflow-x:auto;margin-top:8px;"></div>
        </div>
        <div class="agy-plinth-card obsidian">
          <h3>Matriz Global Floyd-Warshall O(V³) — Tiempos Mínimos (Horas)</h3>
          <p style="font-size:14px;">Extracto de la matriz 25×25 de caminos mínimos entre las 7 asociaciones pesqueras y los nodos estratégicos de acopio, frío y venta.</p>
          <div id="agy-ed-fw-wrap" style="overflow-x:auto;margin-top:8px;"></div>
        </div>
      </div>
    </article>

    <!-- DIAPOSITIVA 7: SOLUCIÓN PROPUESTA (FLP GREEDY), ESCENARIOS Y TIEMPOS TÉRMICOS -->
    <article id="agy-ed-slide-6" class="agy-ed-slide">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:16px;padding-bottom:24px;border-bottom:1px dashed var(--color-cork-border);margin-bottom:28px;">
        <div>
          <h2>La Solución Propuesta: Localización Óptima FLP y Cumplimiento Térmico</h2>
          <p style="margin-top:6px;">Demostración cuantitativa sobre los 5 escenarios de inversión frigorífica y tiempos de acceso sin frío de las 7 asociaciones.</p>
        </div>
        <div class="agy-cta-row">
          <div class="agy-ctrl-group">
            <button id="agy-ed-dest-pg" class="agy-ctrl-btn active" onclick="agyCambiarDestinoEditorial('PG')">Escenario Local (Pto. Gaitán)</button>
            <button id="agy-ed-dest-bog" class="agy-ctrl-btn" onclick="agyCambiarDestinoEditorial('BOG')">Corredor Nacional (Bogotá)</button>
          </div>
        </div>
      </div>

      <!-- 4 KPIs de la Solución Propuesta -->
      <div class="agy-bento-grid-4">
        <div class="agy-plinth-card highlight-saffron">
          <div class="agy-stat-num">Solo F02</div>
          <div class="agy-stat-title">Configuración Óptima FLP</div>
          <div class="agy-stat-desc">Centro Frío Puerto Gaitán en el casco urbano consolida toda la cuenca sin abrir plantas redundantes.</div>
        </div>
        <div class="agy-plinth-card highlight-iris">
          <div class="agy-stat-num">100<span class="agy-stat-unit">%</span></div>
          <div class="agy-stat-title">Cobertura Sanitaria FAO</div>
          <div class="agy-stat-desc">Las 7 asociaciones pesqueras ingresan a refrigeración en t ≤ 2.20 h (muy por debajo del límite de 4.0 h).</div>
        </div>
        <div class="agy-plinth-card">
          <div class="agy-stat-num">$13.02<span class="agy-stat-unit">M/mes</span></div>
          <div class="agy-stat-title">Costo Operativo Optimizado</div>
          <div class="agy-stat-desc">$13,023,272 COP/mes ($59,080 COP/kg), equilibrando costo fijo de planta y fletes de acarreo.</div>
        </div>
        <div class="agy-plinth-card">
          <div class="agy-stat-num" style="color:#10b981;">$10.0<span class="agy-stat-unit">M/mes</span></div>
          <div class="agy-stat-title">Ahorro Directo Mensual</div>
          <div class="agy-stat-desc">Evita el sobrecosto de operar las 3 plantas frigoríficas simultáneamente ($23,023,272 COP/mes).</div>
        </div>
      </div>

      <div class="agy-bento-grid-2">
        <div class="agy-plinth-card">
          <h3>Comparativa de Costo Mensual y Pérdida por Escenario FLP</h3>
          <p style="font-size:14px;">Evaluación técnica de los 5 escenarios reales mediante EvaluadorEscenarios y SimuladorCadenaFrio.</p>
          <div id="agy-ed-flp-chart" style="margin-top:10px;"></div>
        </div>
        <div class="agy-plinth-card">
          <h3>Tiempo Acumulado Sin Frío por Asociación vs. Límite FAO (240 min)</h3>
          <p style="font-size:14px;">Haz clic sobre cualquier asociación para abrir su itinerario óptimo directamente en el Grafo Multimodal.</p>
          <div id="agy-ed-thermal-chart" style="margin-top:10px;"></div>
        </div>
      </div>
    </article>

    <!-- DIAPOSITIVA 8: CONCLUSIONES, ANTES VS DESPUÉS E IMPACTO REGIONAL -->
    <article id="agy-ed-slide-7" class="agy-ed-slide">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:16px;padding-bottom:24px;border-bottom:1px dashed var(--color-cork-border);margin-bottom:28px;">
        <div>
          <h2>Validación Final, Conclusiones e Impacto Socioeconómico</h2>
          <p style="margin-top:6px;">Síntesis de ingeniería de estructuras de datos y valor público entregado al corredor pesquero de Puerto Gaitán, Meta.</p>
        </div>
        <div class="agy-cta-row">
          <button class="agy-pill-cta" onclick="agyCambiarSlide(4, false)">Ir al Grafo Mixto Interactivo</button>
          <button class="agy-pill-ghost" onclick="agyCambiarSlide(0, false)">Volver a Portada</button>
        </div>
      </div>

      <div class="agy-bento-grid-2">
        <div class="agy-plinth-card highlight-iris">
          <h3>Demostración Irrefutable: Antes vs. Con el Modelo Optimizado</h3>
          <table class="agy-table" style="margin-top:10px;">
            <thead>
              <tr>
                <th>Indicador Crítico</th>
                <th>Antes (Sin Frío)</th>
                <th>Modelo Optimizado G=(V,E)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Pérdida poscaptura regional</td>
                <td style="color:var(--color-ember-accent);">Hasta 40.0% (36.1% sim.)</td>
                <td style="color:#10b981;font-weight:600;">1.3% local / 2.1% a Bogotá</td>
              </tr>
              <tr>
                <td>Cumplimiento sanitario (t ≤ 4.0 h)</td>
                <td style="color:var(--color-ember-accent);">0.0% (Riesgo biológico)</td>
                <td style="color:#10b981;font-weight:600;">100.0% (7/7 asociaciones)</td>
              </tr>
              <tr>
                <td>Tiempo máximo expuesto al sol</td>
                <td style="color:var(--color-ember-accent);">> 9.5 h hasta mercado</td>
                <td style="color:var(--color-saffron-spark);font-weight:600;">Máx. 2.20 h (132 min en P05)</td>
              </tr>
              <tr>
                <td>Inversión operativa mensual FLP</td>
                <td>Sin infraestructura</td>
                <td style="color:var(--color-warm-cream);font-weight:600;">$13.02M/mes (Ahorra $10M vs 3 CF)</td>
              </tr>
              <tr>
                <td>Protección de hábitats (H01, H02)</td>
                <td>Tránsito sin control</td>
                <td style="color:#10b981;font-weight:600;">100% excluidos como paso intermedio</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="agy-plinth-card highlight-saffron">
          <h3>Rigor en Estructuras de Datos y Verificación Automatizada</h3>
          <div class="agy-prop-row"><span class="agy-prop-key">Representación del Grafo</span><span class="agy-prop-val">Lista de Adyacencia Hash O(|V|+|E|)</span></div>
          <div class="agy-prop-row"><span class="agy-prop-key">Catálogo Topológico</span><span class="agy-prop-val">25 Vértices (7 tipos) · 41 Aristas (22 vars)</span></div>
          <div class="agy-prop-row"><span class="agy-prop-key">Grafo Fluvial IGAC (rios.shp)</span><span class="agy-prop-val">12,413 V · 24,790 E · 447 celdas Hash</span></div>
          <div class="agy-prop-row"><span class="agy-prop-key">Geometría Vial OSRM</span><span class="agy-prop-val">41 polilíneas reales en caché offline</span></div>
          <div class="agy-prop-row"><span class="agy-prop-key">Algoritmos Propios</span><span class="agy-prop-val">Dijkstra, Bellman-Ford, Floyd-Warshall, BFS, DFS, FLP</span></div>
          <div class="agy-prop-row"><span class="agy-prop-key">Suite de Pruebas Pytest</span><span class="agy-prop-val" style="color:#10b981;">38 / 38 pruebas unitarias 100% verdes</span></div>
        </div>
      </div>

      <div class="agy-bento-grid-3">
        <div class="agy-plinth-card obsidian">
          <div class="agy-stat-title" style="color:var(--color-saffron-spark);">1. Seguridad Alimentaria Regional</div>
          <p style="font-size:14px;">Garantiza que los 265.8 kg/día capturados por los pescadores artesanales de Puerto Gaitán lleguen inocuos a Villavicencio y Bogotá.</p>
        </div>
        <div class="agy-plinth-card obsidian">
          <div class="agy-stat-title" style="color:#10b981;">2. Sostenibilidad Ambiental</div>
          <p style="font-size:14px;">El parámetro algorítmico evitar_restringidos=True desvía automáticamente las embarcaciones fuera de los hábitats de desove H01 y H02.</p>
        </div>
        <div class="agy-plinth-card obsidian">
          <div class="agy-stat-title" style="color:var(--color-electric-link);">3. Escalabilidad Computacional</div>
          <p style="font-size:14px;">Arquitectura modular 100% autónoma y reproducible para planificar infraestructura agroindustrial en toda la cuenca de la Orinoquía.</p>
        </div>
      </div>
    </article>
  </section>

  <!-- ─────────────────────────────────────────────────────────────────────
       BLOQUE B: EXPLORADOR INTERACTIVO DE MAPA SIG Y GRAFOS (SLIDES 2, 3, 4, 5)
       Estética Dala: Constellation floating on black velvet (#000000)
       ───────────────────────────────────────────────────────────────────── -->
  <section id="agy-interactive-stage" aria-label="Explorador Interactivo de Grafos y Mapa SIG">
    <div class="agy-deck-header">
      <div class="agy-deck-title-box">
        <h2>
          <span id="agy-slide-tag" class="agy-section-tag">CONSTELACIÓN G(V,E)</span>
          <span id="agy-slide-title">Explorador Interactivo</span>
        </h2>
        <p id="agy-slide-subtitle">Haz clic sobre cualquier vértice o arista para inspeccionar sus propiedades en memoria.</p>
      </div>

      <div class="agy-deck-controls">
        <div id="agy-layout-switcher" class="agy-ctrl-group">
          <button id="agy-layout-hier" class="agy-ctrl-btn active" onclick="agyCambiarLayout('hier')">
            Vista Jerárquica (Etapas)
          </button>
          <button id="agy-layout-geo" class="agy-ctrl-btn" onclick="agyCambiarLayout('geo')">
            Vista Geográfica (Lupa Focal)
          </button>
        </div>

        <div class="agy-ctrl-group" title="Seleccionar mercado objetivo para itinerarios y escenarios">
          <button id="agy-dest-bog" class="agy-ctrl-btn active" onclick="agyCambiarDestino('BOG')">
            Destino: Bogotá
          </button>
          <button id="agy-dest-pg" class="agy-ctrl-btn" onclick="agyCambiarDestino('PG')">
            Destino: Pto. Gaitán
          </button>
        </div>

        <select id="agy-itin-select" class="agy-ctrl-select" aria-label="Resaltar itinerario de pesca" onchange="agySeleccionarItinerario(this.value)">
          <option value="">Resaltar Itinerario Óptimo...</option>
        </select>

        <select id="agy-quick-select" class="agy-ctrl-select" aria-label="Ir a vértice o arista" onchange="agySeleccionarElementoRapido(this.value)">
          <option value="">Inspeccionar Vértice / Arista...</option>
        </select>

        <button class="agy-ctrl-btn" onclick="agyMostrarComparadorEscenarios()">
          Escenarios FLP
        </button>

        <button class="agy-ctrl-btn" onclick="agyResetVista()" title="Restaurar zoom y encuadre (Tecla R)">
          Centrar Vista
        </button>
      </div>
    </div>

    <div class="agy-deck-body">
      <div class="agy-canvas-wrap" id="agy-canvas-wrap">
        <!-- Partículas Triangulares Dala en Segundo Plano sobre #000000 Void -->
        <canvas id="agy-constellation-canvas" aria-hidden="true"></canvas>

        <!-- Tarjetas KPI en vivo -->
        <div class="agy-kpi-strip" id="agy-kpi-strip"></div>

        <!-- HUD de Coordenadas y Escala -->
        <div id="agy-coords-hud">
          <span id="agy-hud-pos">LAT 4.3130°N · LON 72.0830°W</span>
          <span id="agy-hud-zoom">ZOOM 1.0×</span>
        </div>

        <!-- Panel de Capas y Encuadres SIG (Diapositiva 2: Mapa Geográfico) -->
        <aside id="agy-gis-layers-panel" aria-label="Capas del Mapa Geográfico">
          <div class="agy-gis-panel-title">
            <span>Capas Geográficas SIG</span>
            <span style="color:var(--color-saffron-spark)">OFFLINE</span>
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
            <span><input type="checkbox" id="chk-layer-navrivers" checked onchange="agyToggleCapaGIS()"> Rutas Navegables</span>
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
            <button class="agy-cam-btn" onclick="agyEnfocarRegionGIS('all')">Meta – Bogotá</button>
            <button class="agy-cam-btn" onclick="agyEnfocarRegionGIS('cuenca')">Cuenca Pto. Gaitán</button>
            <button class="agy-cam-btn" onclick="agyEnfocarRegionGIS('urbano')">Casco Urbano</button>
            <button class="agy-cam-btn" onclick="agyEnfocarRegionGIS('rutameta')">Corredor Ruta Meta</button>
          </div>
          <button id="agy-btn-folium-native" class="agy-cam-btn" style="width:100%;margin-top:8px;text-align:center;color:var(--color-saffron-spark);border-color:rgba(255,184,41,0.4);display:none;" onclick="agyAlternarFoliumNativo()">
            Ver Mapa Satelital / Folium Base
          </button>
        </aside>

        <!-- Lienzo SVG Principal de Grafos y SIG -->
        <svg id="agy-graph-svg" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Visualización interactiva de red logística">
          <defs>
            <marker id="arrow-terrestre" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
              <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#15846e"/>
            </marker>
            <marker id="arrow-regular" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
              <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#ffb829"/>
            </marker>
            <marker id="arrow-mala" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
              <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#dc5000"/>
            </marker>
            <marker id="arrow-acceso" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
              <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#9a9a9a"/>
            </marker>
            <marker id="arrow-fluvial" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
              <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#2997ff"/>
            </marker>
            <marker id="arrow-active" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
              <path d="M 0 1 L 10 5 L 0 9 z" fill="#ffb829"/>
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
          <span style="color:var(--color-steel)">|</span>
          <span class="agy-filter-pill" style="cursor:default"><span class="agy-leg-line" style="border-top:2.5px dashed #2997ff"></span>Fluvial</span>
          <span class="agy-filter-pill" style="cursor:default"><span class="agy-leg-line" style="border-top:2.5px solid #9a9a9a"></span>Acceso Local</span>
          <span class="agy-filter-pill" style="cursor:default"><span class="agy-leg-line" style="border-top:2.5px solid #15846e"></span>Terrestre</span>
        </div>
      </div>

      <!-- Inspector Técnico de Datos (Derecha) -->
      <aside class="agy-inspector" aria-label="Inspector de propiedades del elemento seleccionado">
        <div class="agy-inspector-header">
          <h3 id="agy-insp-title">Inspector de Estructura de Datos</h3>
          <span id="agy-insp-badge" class="agy-badge" style="background:var(--color-void);color:var(--color-saffron-spark);">G=(V,E)</span>
        </div>
        <div class="agy-inspector-content" id="agy-insp-body"></div>
      </aside>
    </div>
  </section>

  <!-- ─────────────────────────────────────────────────────────────────────
       BLOQUE C: MAPA INTERACTIVO FOLIUM ORIGINAL (DIAPOSITIVA 2 EN MODO AUTÓNOMO)
       ───────────────────────────────────────────────────────────────────── -->
  <section id="agy-folium-stage" aria-label="Mapa Geográfico Interactivo Folium">
    <iframe id="agy-folium-iframe" src="mapa_interactivo.html?folium_only=1" title="Mapa Geográfico Folium — Cadena de Frío Puerto Gaitán"></iframe>
  </section>
</main>

<script>
const AGY_DATA = {json_payload};
const AGY_STANDALONE = {"true" if modo_autonomo else "false"};

const AGY_COLORES_NODO = {{
  P: "#2997ff",
  M: "#8052ff",
  A: "#ffb829",
  F: "#15846e",
  D: "#dc5000",
  H: "#10b981",
  V: "#ec4899"
}};

const AGY_ETAPAS_JERARQUICAS = [
  {{ col: 0, titulo: "1. CAPTURA (P)" }},
  {{ col: 1, titulo: "2. TRANSBORDO (M)" }},
  {{ col: 2, titulo: "3. ACOPIO (A)" }},
  {{ col: 3, titulo: "4. CONTROL (D / H)" }},
  {{ col: 4, titulo: "5. FRÍO (F)" }},
  {{ col: 5, titulo: "6. VENTA (V)" }}
];

// Estado global de las 8 Diapositivas:
// 0: Portada & Problema (Editorial)
// 1: Mapa Geográfico SIG (Interactivo)
// 2: Grafo Vial (Interactivo)
// 3: Grafo Fluvial (Interactivo)
// 4: Grafo Mixto Multimodal (Interactivo)
// 5: Núcleo Algorítmico & Defensa (Editorial)
// 6: Solución FLP & KPIs Térmicos (Editorial)
// 7: Conclusiones & Impacto (Editorial)
let agySlideActual = 0;
let agyLayoutModo = "hier";
let agyDestinoModo = "BOG";
let agyFiltroTipoNodo = "ALL";
let agySeleccion = null;
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
let agyFoliumNativoVisible = false;

let agyDomNodes = {{}};
let agyDomEdges = {{}};

function agyPrefersReducedMotion() {{
  return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}}

function agyEsSlideInteractivo(s) {{
  return s === 1 || s === 2 || s === 3 || s === 4;
}}

// ═══════════════════════════════════════════════════════════════════════
// MOTOR DE CONSTELACIÓN DE PARTÍCULAS TRIANGULARES (ESTÉTICA DALA)
// ═══════════════════════════════════════════════════════════════════════
function agyIniciarConstelacionDala() {{
  const palette = ["#8052ff", "#ffb829", "#15846e", "#2997ff", "#ec4899"];

  // 1. Lienzo de fondo del Explorador de Grafos (#agy-constellation-canvas)
  const bgCanvas = document.getElementById("agy-constellation-canvas");
  if (bgCanvas) {{
    const ctx = bgCanvas.getContext("2d");
    const particles = Array.from({{ length: 78 }}, (_, i) => ({{
      x: (Math.sin(i * 12.9898) * 43758.5453) % 1,
      y: (Math.cos(i * 78.233) * 19341.231) % 1,
      size: 2.5 + (i % 4) * 1.1,
      angle: (i * 0.65) % (Math.PI * 2),
      color: palette[i % palette.length],
      alpha: 0.16 + (i % 5) * 0.06
    }}));

    function drawBgConstellation() {{
      const w = bgCanvas.clientWidth || 960;
      const h = bgCanvas.clientHeight || 640;
      if (bgCanvas.width !== w || bgCanvas.height !== h) {{
        bgCanvas.width = w;
        bgCanvas.height = h;
      }}
      ctx.clearRect(0, 0, w, h);
      particles.forEach((pt, idx) => {{
        const px = Math.abs(pt.x) * w;
        const py = Math.abs(pt.y) * h;
        ctx.save();
        ctx.translate(px, py);
        ctx.rotate(pt.angle);
        ctx.strokeStyle = pt.color;
        ctx.globalAlpha = pt.alpha;
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        const r = pt.size;
        ctx.moveTo(0, -r);
        ctx.lineTo(r * 0.866, r * 0.5);
        ctx.lineTo(-r * 0.866, r * 0.5);
        ctx.closePath();
        ctx.stroke();
        ctx.restore();
      }});
    }}
    drawBgConstellation();
    window.addEventListener("resize", drawBgConstellation);
  }}

  // 2. Constelación Neural/Topológica en la Portada (#agy-hero-constellation)
  const heroCanvas = document.getElementById("agy-hero-constellation");
  if (heroCanvas) {{
    const hCtx = heroCanvas.getContext("2d");
    const triNodes = (AGY_DATA.nodos || []).map((n, i) => {{
      const ang = (i / 25) * Math.PI * 2;
      const rad = 0.24 + (i % 3) * 0.09;
      return {{
        id: n.id,
        tipo: n.tipo,
        bx: 0.5 + Math.cos(ang) * rad * 1.15,
        by: 0.46 + Math.sin(ang) * rad * 0.85,
        color: AGY_COLORES_NODO[n.tipo] || "#8052ff",
        phase: i * 0.42
      }};
    }});
    const dust = Array.from({{ length: 120 }}, (_, i) => {{
      const a = (i * 2.39996);
      const r = Math.sqrt(i / 120) * 0.42;
      return {{
        x: 0.5 + Math.cos(a) * r * 1.2,
        y: 0.46 + Math.sin(a) * r * 0.88,
        sz: 2.0 + (i % 3) * 1.1,
        rot: i * 0.5,
        col: palette[i % palette.length],
        alpha: 0.28 + (i % 4) * 0.14
      }};
    }});

    let mouseOff = {{ x: 0, y: 0 }};
    heroCanvas.addEventListener("mousemove", (ev) => {{
      const r = heroCanvas.getBoundingClientRect();
      mouseOff.x = ((ev.clientX - r.left) / (r.width || 1) - 0.5) * 18;
      mouseOff.y = ((ev.clientY - r.top) / (r.height || 1) - 0.5) * 18;
      renderHeroFrame(performance.now() * 0.001);
    }});
    heroCanvas.addEventListener("click", () => agyCambiarSlide(4, false));

    function renderHeroFrame(t) {{
      const w = heroCanvas.clientWidth || 480;
      const h = heroCanvas.clientHeight || 390;
      if (heroCanvas.width !== w || heroCanvas.height !== h) {{
        heroCanvas.width = w;
        heroCanvas.height = h;
      }}
      hCtx.clearRect(0, 0, w, h);

      // Conexiones tenues de la constelación
      hCtx.strokeStyle = "rgba(128, 82, 255, 0.18)";
      hCtx.lineWidth = 0.9;
      for (let i = 0; i < triNodes.length; i++) {{
        const a = triNodes[i];
        const b = triNodes[(i + 3) % triNodes.length];
        hCtx.beginPath();
        hCtx.moveTo(a.bx * w + mouseOff.x * 0.6, a.by * h + mouseOff.y * 0.6);
        hCtx.lineTo(b.bx * w + mouseOff.x * 0.6, b.by * h + mouseOff.y * 0.6);
        hCtx.stroke();
      }}

      // Nube de triángulos Dala
      dust.forEach((d, idx) => {{
        const px = d.x * w + mouseOff.x * ((idx % 3) * 0.35);
        const py = d.y * h + mouseOff.y * ((idx % 3) * 0.35);
        hCtx.save();
        hCtx.translate(px, py);
        hCtx.rotate(d.rot + t * 0.15);
        hCtx.strokeStyle = d.col;
        hCtx.globalAlpha = d.alpha;
        hCtx.lineWidth = 1.15;
        hCtx.beginPath();
        const s = d.sz;
        hCtx.moveTo(0, -s);
        hCtx.lineTo(s * 0.866, s * 0.5);
        hCtx.lineTo(-s * 0.866, s * 0.5);
        hCtx.closePath();
        hCtx.stroke();
        hCtx.restore();
      }});

      // Vértices principales con etiqueta
      triNodes.forEach((nd) => {{
        const px = nd.bx * w + mouseOff.x;
        const py = nd.by * h + mouseOff.y;
        hCtx.save();
        hCtx.translate(px, py);
        hCtx.fillStyle = "#000000";
        hCtx.strokeStyle = nd.color;
        hCtx.lineWidth = 1.8;
        hCtx.beginPath();
        hCtx.arc(0, 0, 5.5, 0, Math.PI * 2);
        hCtx.fill();
        hCtx.stroke();
        hCtx.fillStyle = "#f5f5f7";
        hCtx.font = "600 9px monospace";
        hCtx.fillText(nd.id, 8, 3);
        hCtx.restore();
      }});
    }}
    renderHeroFrame(0);
    window.addEventListener("resize", () => renderHeroFrame(0));
  }}
}}

// ═══════════════════════════════════════════════════════════════════════
// RENDERIZADO DE WIDGETS EDITORIALES (DIAPOSITIVAS 6 Y 7)
// ═══════════════════════════════════════════════════════════════════════
function agyInicializarSlidesEditoriales() {{
  // 1. Selector del simulador multicriterio (Diapositiva 6)
  const selEdge = document.getElementById("agy-ed-edge-sim");
  if (selEdge && AGY_DATA.aristas.length) {{
    selEdge.innerHTML = AGY_DATA.aristas.map(a =>
      `<option value="${{a.id}}">#${{String(a.idx).padStart(2, '0')}} ${{a.origen}} → ${{a.destino}} (${{a.tipo_via}}, ${{a.distancia_km}} km, w=${{a.peso_combinado.toFixed(4)}})</option>`
    ).join("");
    agyActualizarSimuladorMulticriterio(AGY_DATA.aristas[0].id);
  }}

  // 2. Tabla de Verificación Cruzada Bellman-Ford vs Dijkstra (Diapositiva 6)
  const bfWrap = document.getElementById("agy-ed-bf-table-wrap");
  if (bfWrap) {{
    const filasBF = (AGY_DATA.verificacion_bf || []).map(r => `
      <tr>
        <td>P01 → ${{r.destino}} (${{r.nombre_destino}})</td>
        <td style="text-align:right">${{r.dijkstra_h.toFixed(2)}} h</td>
        <td style="text-align:right">${{r.bellman_ford_h.toFixed(2)}} h</td>
        <td style="text-align:right;color:#10b981;font-weight:600;">${{r.coincide ? "EXACTO" : "DIFF"}}</td>
      </tr>
    `).join("");
    bfWrap.innerHTML = `
      <table class="agy-table">
        <thead>
          <tr>
            <th>Consulta (evitar_restringidos=True)</th>
            <th style="text-align:right">Dijkstra</th>
            <th style="text-align:right">Bellman-Ford</th>
            <th style="text-align:right">Validación</th>
          </tr>
        </thead>
        <tbody>${{filasBF}}</tbody>
      </table>
      <div style="margin-top:8px;font-family:var(--font-mono);font-size:12px;color:var(--color-warm-cream);">
        Detección de ciclos negativos: <strong style="color:#10b981;">tiene_ciclo_negativo = ${{AGY_DATA.tiene_ciclo_negativo ? "True" : "False"}} (Grafo consistente)</strong>
      </div>
    `;
  }}

  // 3. Tabla de Recorridos BFS (Saltos Mínimos) y DFS (Rutas Alternativas) (Diapositiva 6)
  const bfsWrap = document.getElementById("agy-ed-bfs-dfs-wrap");
  if (bfsWrap) {{
    const dfsMap = {{}};
    (AGY_DATA.dfs_rutas || []).forEach(d => {{ dfsMap[d.origen_id] = d; }});
    const filasBFS = (AGY_DATA.bfs_saltos || []).map(b => {{
      const dInfo = dfsMap[b.origen_id] || {{ num_rutas: 1 }};
      return `
        <tr style="cursor:pointer" onclick="agySaltarAItinerario('${{b.origen_id}}')">
          <td><strong>${{b.origen_id}}</strong> · ${{agyNombreCompacto(b.nombre)}}</td>
          <td style="text-align:right;color:var(--color-saffron-spark);font-weight:600;">${{b.saltos}} saltos</td>
          <td>${{b.camino.join(" → ")}}</td>
          <td style="text-align:right">${{dInfo.num_rutas}} rutas</td>
        </tr>
      `;
    }}).join("");
    bfsWrap.innerHTML = `
      <table class="agy-table">
        <thead>
          <tr>
            <th>Asociación (P)</th>
            <th style="text-align:right">BFS Saltos</th>
            <th>Camino de Saltos Mínimos a F02</th>
            <th style="text-align:right">DFS Simples</th>
          </tr>
        </thead>
        <tbody>${{filasBFS}}</tbody>
      </table>
    `;
  }}

  // 4. Matriz Floyd-Warshall O(V³) (Diapositiva 6)
  const fwWrap = document.getElementById("agy-ed-fw-wrap");
  if (fwWrap) {{
    const cols = ["A01", "A02", "M03", "F01", "F02", "F03", "V_PG1", "V_BOG1"];
    const ths = cols.map(c => `<th style="text-align:right">${{c}}</th>`).join("");
    const trs = (AGY_DATA.matriz_fw || []).map(row => {{
      const tds = cols.map(c => {{
        const cell = row.destinos[c];
        const val = cell && cell.tiempo_h !== null ? `${{cell.tiempo_h.toFixed(2)}}${{c === 'F02' ? '★' : ''}}` : "∞";
        const colStyle = c === "F02" ? "color:var(--color-saffron-spark);font-weight:600;" : "";
        return `<td style="text-align:right;${{colStyle}}">${{val}}</td>`;
      }}).join("");
      return `<tr><td><strong>${{row.origen_id}}</strong></td>${{tds}}</tr>`;
    }}).join("");
    fwWrap.innerHTML = `
      <table class="agy-table">
        <thead><tr><th>Origen</th>${{ths}}</tr></thead>
        <tbody>${{trs}}</tbody>
      </table>
    `;
  }}

  // 5. Gráficos de FLP y Tiempos Térmicos (Diapositiva 7)
  agyRenderizarGraficosSlide7("PG");
}}

function agyActualizarSimuladorMulticriterio(edgeId) {{
  const a = (AGY_DATA.aristas || []).find(x => x.id === edgeId) || AGY_DATA.aristas[0];
  const box = document.getElementById("agy-ed-multicriterio-box");
  if (!a || !box) return;
  const w = a.peso_combinado || 0.0001;
  const d = a.desglose_peso;
  const pct = (v) => Math.min(100, Math.round((v / w) * 100));

  box.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
      <h3>Desglose de Arista ${{a.origen}} → ${{a.destino}} (${{a.tipo_via}})</h3>
      <span class="agy-badge" style="color:var(--color-saffron-spark);">w_ij = ${{a.peso_combinado.toFixed(4)}}</span>
    </div>
    <p style="font-size:14px;">Distancia: ${{a.distancia_km}} km · Tiempo efectivo: ${{a.tiempo_efectivo_h.toFixed(2)}} h · Costo total: $${{a.costo_total_cop.toLocaleString()}} COP · Estado: ${{a.estado_via}}</p>
    <div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span>α=0.25 · Tiempo efectivo (${{a.tiempo_efectivo_h.toFixed(2)}} h)</span><span>+${{d.t_term.toFixed(4)}} (${{pct(d.t_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.t_term)}}%"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span>β=0.20 · Costo total ($${{a.costo_total_cop.toLocaleString()}} COP)</span><span>+${{d.c_term.toFixed(4)}} (${{pct(d.c_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.c_term)}}%;background:#2997ff;"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span>γ=0.20 · Riesgo combinado (${{a.riesgo_combinado.toFixed(3)}})</span><span>+${{d.r_term.toFixed(4)}} (${{pct(d.r_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.r_term)}}%;background:#ffb829;"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span>δ=0.10 · Pérdida térmica (${{a.perdida_esperada.toFixed(2)}}%)</span><span>+${{d.p_term.toFixed(4)}} (${{pct(d.p_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.p_term)}}%;background:#dc5000;"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span>ε=0.10 · Incertidumbre vial (1 - ${{a.confiabilidad_via.toFixed(2)}})</span><span>+${{d.conf_term.toFixed(4)}} (${{pct(d.conf_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.conf_term)}}%;background:#15846e;"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span>ζ=0.05 · Probabilidad de retraso (${{a.prob_retraso.toFixed(2)}})</span><span>+${{d.ret_term.toFixed(4)}} (${{pct(d.ret_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.ret_term)}}%;background:#ec4899;"></div></div>
      </div>
    </div>
  `;
}}

function agyCambiarDestinoEditorial(modo) {{
  document.getElementById("agy-ed-dest-pg")?.classList.toggle("active", modo === "PG");
  document.getElementById("agy-ed-dest-bog")?.classList.toggle("active", modo === "BOG");
  agyRenderizarGraficosSlide7(modo);
}}

function agyRenderizarGraficosSlide7(modo) {{
  const flpWrap = document.getElementById("agy-ed-flp-chart");
  const thWrap = document.getElementById("agy-ed-thermal-chart");
  const listaEsc = modo === "PG" ? (AGY_DATA.escenarios_pg || []) : (AGY_DATA.escenarios_bog || []);

  // Costos mensuales de referencia FLP + métricas reales del simulador
  const costosMensualesRef = [
    {{ label: "1. Sin Cadena de Frío", mensualM: 8.00, optimo: false }},
    {{ label: "2. Solo F01 (Alto Neblinas)", mensualM: 9.70, optimo: false }},
    {{ label: "3. Solo F02 (Puerto Gaitán) ★ ÓPTIMO FLP", mensualM: 13.02, optimo: true }},
    {{ label: "4. Solo F03 (Puerto López)", mensualM: 13.02, optimo: false }},
    {{ label: "5. 3 Centros (F01 + F02 + F03)", mensualM: 23.02, optimo: false }}
  ];

  if (flpWrap) {{
    const barrasCostos = costosMensualesRef.map((c, idx) => {{
      const escSim = listaEsc[idx] || {{ pct_perdida: 0, costo_por_kg_cop: 0, cumplimiento_termico_pct: 100 }};
      const wPct = Math.round((c.mensualM / 23.02) * 100);
      const colBar = c.optimo ? "var(--color-saffron-spark)" : (idx === 0 ? "var(--color-ember-accent)" : "var(--color-electric-iris)");
      return `
        <div class="agy-bar-wrap" style="margin:10px 0;">
          <div class="agy-bar-label">
            <span style="${{c.optimo ? 'color:var(--color-saffron-spark);font-weight:700;' : ''}}">${{c.label}}</span>
            <span>$${{c.mensualM.toFixed(2)}}M/mes · Pérdida: ${{escSim.pct_perdida.toFixed(1)}}%</span>
          </div>
          <div class="agy-bar-bg" style="height:10px;">
            <div class="agy-bar-fill" style="width:${{wPct}}%;background:${{colBar}};"></div>
          </div>
        </div>
      `;
    }}).join("");

    const filasSim = listaEsc.map((e, i) => `
      <tr class="${{i === 2 ? 'row-optimal' : ''}}">
        <td>${{e.nombre}} ${{i === 2 ? '★' : ''}}</td>
        <td style="text-align:right">${{e.pct_perdida.toFixed(1)}}%</td>
        <td style="text-align:right">${{e.kg_perdido.toFixed(1)}} kg</td>
        <td style="text-align:right">$${{e.costo_por_kg_cop.toLocaleString()}}</td>
      </tr>
    `).join("");

    flpWrap.innerHTML = `
      ${{barrasCostos}}
      <table class="agy-table" style="margin-top:14px;">
        <thead>
          <tr>
            <th>Escenario Simulado (${{modo === 'PG' ? 'Pto. Gaitán' : 'Bogotá'}})</th>
            <th style="text-align:right">% Pérd.</th>
            <th style="text-align:right">Merma</th>
            <th style="text-align:right">Costo/kg</th>
          </tr>
        </thead>
        <tbody>${{filasSim}}</tbody>
      </table>
    `;
  }}

  if (thWrap) {{
    const limiteMin = 240;
    const barrasTiempos = (AGY_DATA.tiempos_acceso_frio || []).map(t => {{
      const wPct = Math.min(100, Math.max(4, Math.round((t.tiempo_min / limiteMin) * 100)));
      const col = t.tiempo_min <= 60 ? "#10b981" : "var(--color-saffron-spark)";
      return `
        <div class="agy-bar-wrap" style="margin:10px 0;cursor:pointer;" onclick="agySaltarAItinerario('${{t.origen_id}}')">
          <div class="agy-bar-label">
            <span><strong>${{t.origen_id}}</strong> · ${{agyNombreCompacto(t.nombre)}} (${{t.camino_frio.join(" → ")}})</span>
            <span style="color:${{col}};font-weight:600;">${{t.tiempo_min}} min (${{t.tiempo_h.toFixed(2)}} h)</span>
          </div>
          <div class="agy-bar-bg" style="height:10px;position:relative;">
            <div class="agy-bar-fill" style="width:${{wPct}}%;background:${{col}};"></div>
          </div>
        </div>
      `;
    }}).join("");

    thWrap.innerHTML = `
      <div style="display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:12px;color:var(--color-ash-gray);margin-bottom:8px;">
        <span>Meta Operativa Excelente: &lt; 60 min (1.0 h)</span>
        <span style="color:var(--color-saffron-spark);">Límite Máximo FAO: 240 min (4.0 h)</span>
      </div>
      ${{barrasTiempos}}
      <div class="agy-effect-box" style="margin-top:14px;margin-bottom:0;">
        <strong>Cumplimiento Regional 100%:</strong> Incluso la cuenca más distante (P05 Puerto Trujillo en el río Iteviare, 63 km) alcanza la cadena de frío en <strong>132 min (2.20 h)</strong>, preservando un margen de seguridad de 108 minutos respecto al límite sanitario FAO.
      </div>
    `;
  }}
}}

function agySaltarAItinerario(origenId) {{
  agyCambiarSlide(4, false);
  const selIt = document.getElementById("agy-itin-select");
  if (selIt) selIt.value = origenId;
  agySeleccionarItinerario(origenId);
}}

// ═══════════════════════════════════════════════════════════════════════
// MOTOR INTERACTIVO DE GRAFOS Y MAPA SIG (SLIDES 1, 2, 3, 4 -> VISTAS 2..5)
// ═══════════════════════════════════════════════════════════════════════
function agyListaItinerariosActiva() {{
  return agyDestinoModo === "PG" ? (AGY_DATA.itinerarios_locales || []) : (AGY_DATA.itinerarios || []);
}}

function agyPoblarSelectorItinerarios() {{
  const selIt = document.getElementById("agy-itin-select");
  if (!selIt) return;
  const valorPrevio = selIt.value;
  selIt.innerHTML = `<option value="">Resaltar Itinerario (${{agyDestinoModo === "PG" ? "Pto. Gaitán" : "Bogotá"}})...</option>`;
  agyListaItinerariosActiva().forEach(it => {{
    const opt = document.createElement("option");
    opt.value = it.origen_id;
    opt.textContent = `${{it.origen_id}} → ${{it.destino_id}}: ${{it.ruta.join(" → ")}} (${{it.distancia_total_km}} km)`;
    selIt.appendChild(opt);
  }});
  if (valorPrevio) selIt.value = valorPrevio;
}}

function agyCambiarDestino(modo) {{
  agyDestinoModo = modo;
  document.getElementById("agy-dest-bog")?.classList.toggle("active", modo === "BOG");
  document.getElementById("agy-dest-pg")?.classList.toggle("active", modo === "PG");
  agyPoblarSelectorItinerarios();
  if (agyItinerarioActivo) {{
    agySeleccionarItinerario(agyItinerarioActivo);
  }} else {{
    agyRenderizarEscenaCompleta();
  }}
}}

function agySubgrafoActual() {{
  let aristas = [];
  if (agySlideActual === 2) {{
    // Diapositiva 3: Exclusivamente Grafo Vial (terrestre + acceso_local)
    aristas = AGY_DATA.aristas.filter(a => a.tipo_via === "terrestre" || a.tipo_via === "acceso_local");
  }} else if (agySlideActual === 3) {{
    // Diapositiva 4: Exclusivamente Grafo Fluvial (fluvial)
    aristas = AGY_DATA.aristas.filter(a => a.tipo_via === "fluvial");
  }} else {{
    // Diapositiva 2 (Mapa SIG) y Diapositiva 5 (Grafo Mixto Multimodal)
    aristas = AGY_DATA.aristas.slice();
  }}

  const idsActivos = new Set();
  aristas.forEach(a => {{
    idsActivos.add(a.origen);
    idsActivos.add(a.destino);
  }});

  let nodos = AGY_DATA.nodos.filter(n => (agySlideActual === 1 || agySlideActual === 4) || idsActivos.has(n.id));
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

function agyProyectarSIG(lon, lat, W, H) {{
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

// Lupa Focal Geográfica Ampliada: expande ampliamente la zona de Ruta Meta, Deterioro (D01/D02) y Casco Urbano
function agyProyectarGeoLupa(lon, lat, W, H, padX, padY) {{
  const domLon = [-74.10, -72.80, -72.14, -72.085, -72.03, -71.48];
  const rngLon = [0.04,   0.17,   0.26,   0.57,    0.86,   0.96];
  const nx = agyInterpolarTramo(lon, domLon, rngLon);

  const domLat = [4.08, 4.16, 4.24, 4.31, 4.36, 4.70];
  const rngLat = [0.93, 0.77, 0.60, 0.37, 0.17, 0.05];
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
  const padX = 84, padYTop = 114, padYBot = 68;
  const usableH = Math.max(220, H - padYTop - padYBot);

  if (agySlideActual === 1) {{
    // Diapositiva 2: Mapa Geográfico SIG con coordenadas reales WGS84
    AGY_DATA.nodos.forEach((n) => {{
      const p = agyProyectarSIG(n.longitud, n.latitud, W, H);
      agyTargets[n.id] = {{ x: p.x, y: p.y }};
    }});
    const idsGIS = AGY_DATA.nodos.map(n => n.id);
    const minDistGIS = 5.4;
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
      const ny = (p.y - padYTop) / Math.max(1, (H - 2 * padYTop));
      const yMapped = padYTop + Math.max(0, Math.min(1, ny)) * usableH;
      const rol = ordenTipo[n.tipo] !== undefined ? ordenTipo[n.tipo] : 3;
      const ang = (idx * 2.39996) + rol * 0.55;
      basePos[n.id] = {{
        x: p.x + (rol - 2.5) * 18 + Math.cos(ang) * 14,
        y: yMapped + Math.sin(ang) * 16
      }};
      agyTargets[n.id] = {{ x: basePos[n.id].x, y: basePos[n.id].y }};
    }});

    const minSepX = Math.max(122, Math.min(152, W * 0.13));
    const minSepY = Math.max(66, Math.min(86, usableH * 0.128));
    const ids = AGY_DATA.nodos.map(n => n.id);

    for (let iter = 0; iter < 190; iter++) {{
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
            const push = (1.0 - dEllipse) * 0.55;
            const ux = nx / (dEllipse || 1);
            const uy = ny / (dEllipse || 1);
            a.x -= ux * minSepX * push;
            a.y -= uy * minSepY * push;
            b.x += ux * minSepX * push;
            b.y += uy * minSepY * push;
          }}
        }}
      }}
      const kAnchor = 0.026;
      ids.forEach(id => {{
        const t = agyTargets[id];
        const b = basePos[id];
        t.x += (b.x - t.x) * kAnchor;
        t.y += (b.y - t.y) * kAnchor;
        t.x = Math.max(padX - 16, Math.min(W - padX + 16, t.x));
        t.y = Math.max(padYTop, Math.min(H - padYBot, t.y));
      }});
    }}
  }} else {{
    const colMap = {{ P: 0, M: 1, A: 2, D: 3, H: 3, F: 4, V: 5 }};
    const grupos = {{ 0: [], 1: [], 2: [], 3: [], 4: [], 5: [] }};
    AGY_DATA.nodos.forEach(n => {{
      const c = colMap[n.tipo] !== undefined ? colMap[n.tipo] : 3;
      grupos[c].push(n);
    }});

    Object.keys(grupos).forEach(colStr => {{
      const col = Number(colStr);
      const lista = grupos[col];
      const x = padX + (col / 5) * (W - 2 * padX);
      lista.forEach((n, idx) => {{
        const y = padYTop + ((idx + 0.5) / Math.max(1, lista.length)) * usableH;
        agyTargets[n.id] = {{ x, y }};
      }});
    }});
  }}

  if (inmediato || agyPrefersReducedMotion() || Object.keys(agyPosiciones).length === 0) {{
    AGY_DATA.nodos.forEach(n => {{
      if (agyTargets[n.id]) {{
        agyPosiciones[n.id] = {{ x: agyTargets[n.id].x, y: agyTargets[n.id].y }};
      }}
    }});
  }} else {{
    agyAnimarTransicionPosiciones();
  }}
}}

function agyRutaCurvaArista(p1, p2) {{
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const dist = Math.hypot(dx, dy) || 1;
  const offset = Math.min(26, dist * 0.12);
  const mx = (p1.x + p2.x) / 2 - (dy / dist) * offset;
  const my = (p1.y + p2.y) / 2 + (dx / dist) * offset;
  return {{
    d: `M ${{p1.x.toFixed(1)}} ${{p1.y.toFixed(1)}} Q ${{mx.toFixed(1)}} ${{my.toFixed(1)}} ${{p2.x.toFixed(1)}} ${{p2.y.toFixed(1)}}`,
    lx: 0.25 * p1.x + 0.5 * mx + 0.25 * p2.x,
    ly: 0.25 * p1.y + 0.5 * my + 0.25 * p2.y - 5
  }};
}}

function agyActualizarGeometriaDOM() {{
  Object.keys(agyDomNodes).forEach(id => {{
    const gNode = agyDomNodes[id];
    const pos = agyPosiciones[id];
    if (gNode && pos) {{
      gNode.setAttribute("transform", `translate(${{pos.x.toFixed(1)}}, ${{pos.y.toFixed(1)}})`);
    }}
  }});

  if (agySlideActual === 1) return;

  Object.keys(agyDomEdges).forEach(edgeId => {{
    const ref = agyDomEdges[edgeId];
    const p1 = agyPosiciones[ref.arista.origen];
    const p2 = agyPosiciones[ref.arista.destino];
    if (ref && p1 && p2) {{
      const geom = agyRutaCurvaArista(p1, p2);
      ref.pathVis.setAttribute("d", geom.d);
      ref.pathHit.setAttribute("d", geom.d);
      if (ref.txt) {{
        ref.txt.setAttribute("x", geom.lx.toFixed(1));
        ref.txt.setAttribute("y", geom.ly.toFixed(1));
      }}
    }}
  }});
}}

function agyAnimarTransicionPosiciones() {{
  if (agySpringAnimId) cancelAnimationFrame(agySpringAnimId);
  function step() {{
    let maxDelta = 0;
    AGY_DATA.nodos.forEach(n => {{
      const cur = agyPosiciones[n.id];
      const tgt = agyTargets[n.id];
      if (cur && tgt) {{
        const dx = tgt.x - cur.x;
        const dy = tgt.y - cur.y;
        cur.x += dx * 0.24;
        cur.y += dy * 0.24;
        maxDelta = Math.max(maxDelta, Math.abs(dx), Math.abs(dy));
      }}
    }});
    agyActualizarGeometriaDOM();
    if (maxDelta > 0.45) {{
      agySpringAnimId = requestAnimationFrame(step);
    }} else {{
      AGY_DATA.nodos.forEach(n => {{
        if (agyTargets[n.id]) {{
          agyPosiciones[n.id] = {{ x: agyTargets[n.id].x, y: agyTargets[n.id].y }};
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
  if (agySlideActual === 1) {{
    agyEnfocarRegionGIS("cuenca");
  }} else {{
    agyTransform = {{ x: 0, y: 0, k: 1 }};
    agyActualizarTransform();
    agyCalcularPosicionesObjetivo(false);
  }}
}}

function agyCambiarLayout(modo) {{
  agyLayoutModo = modo;
  document.getElementById("agy-layout-geo")?.classList.toggle("active", modo === "geo");
  document.getElementById("agy-layout-hier")?.classList.toggle("active", modo === "hier");
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
  if (agySlideActual === 1) {{
    agyRenderizarEscenaCompleta(true);
  }}
}}

function agyAlternarFoliumNativo() {{
  if (AGY_STANDALONE) return;
  agyFoliumNativoVisible = !agyFoliumNativoVisible;
  const shell = document.getElementById("agy-app-shell");
  if (agyFoliumNativoVisible) {{
    shell.classList.remove("visible");
  }} else {{
    shell.classList.add("visible");
  }}
}}

function agyCambiarSlide(nuevoSlide, viaKeyboard) {{
  if (nuevoSlide < 0 || nuevoSlide > 7) return;
  agySlideActual = nuevoSlide;
  agyFoliumNativoVisible = false;

  document.getElementById("agy-btn-prev").disabled = (agySlideActual === 0);
  document.getElementById("agy-btn-next").disabled = (agySlideActual === 7);
  document.getElementById("agy-slide-counter").textContent = `Vista ${{agySlideActual + 1}} / 8`;

  document.querySelectorAll(".agy-slide-tab").forEach(btn => {{
    const activo = Number(btn.dataset.slide) === agySlideActual;
    btn.classList.toggle("active", activo);
    btn.setAttribute("aria-selected", activo ? "true" : "false");
  }});

  const shell = document.getElementById("agy-app-shell");
  const edStage = document.getElementById("agy-editorial-stage");
  const intStage = document.getElementById("agy-interactive-stage");
  const foliumStage = document.getElementById("agy-folium-stage");

  if (viaKeyboard) {{
    shell.classList.add("keyboard-instant");
  }} else {{
    shell.classList.remove("keyboard-instant");
  }}

  // En Diapositiva 2 (agySlideActual === 1: Mapa Geográfico), conservar intacto el Mapa Folium original
  const serialEl = document.getElementById("agy-oryzo-serial");
  if (serialEl) serialEl.style.display = (agySlideActual === 1) ? "none" : "";

  if (agySlideActual === 1) {{
    if (!AGY_STANDALONE) {{
      // Dentro de mapa_interactivo.html: revelar el mapa Folium original debajo de la barra superior
      shell.classList.remove("visible");
      return;
    }} else {{
      // Dentro de presentacion_red_logistica.html: mostrar el mapa Folium original vía iframe
      shell.classList.add("visible");
      edStage.classList.remove("active");
      intStage.classList.remove("active");
      if (foliumStage) foliumStage.classList.add("active");
      return;
    }}
  }}

  shell.classList.add("visible");
  if (foliumStage) foliumStage.classList.remove("active");

  if (!agyEsSlideInteractivo(agySlideActual)) {{
    edStage.classList.add("active");
    intStage.classList.remove("active");
    document.querySelectorAll(".agy-ed-slide").forEach(art => art.classList.remove("active"));
    const targetArt = document.getElementById(`agy-ed-slide-${{agySlideActual}}`);
    if (targetArt) targetArt.classList.add("active");
    edStage.scrollTop = 0;
    return;
  }}

  edStage.classList.remove("active");
  intStage.classList.add("active");

  const metaSlides = {{
    1: {{
      tag: "MAPA SIG Y ALCANCE",
      titulo: "Diapositiva 2: Selección de Centros Poblados y Mapa Geográfico SIG",
      sub: "Red hidrográfica IGAC rios.shp (12,413 vértices), trazados viales OSRM, 25 nodos del modelo y 49 puntos de campo."
    }},
    2: {{
      tag: "CONSTELACIÓN VIAL",
      titulo: "Diapositiva 3: Grafo Vial Interactivo (Red Terrestre y Accesos Locales)",
      sub: "Subgrafo exclusivo de 31 aristas terrestres (carreteras nacionales y accesos locales) y 19 vértices sobre lienzo Dala."
    }},
    3: {{
      tag: "CONSTELACIÓN FLUVIAL",
      titulo: "Diapositiva 4: Grafo de Ríos Interactivo (Rutas Navegables vs. Hidrografía)",
      sub: "Subgrafo de 10 rutas de navegación artesanal sobre la malla hidrográfica de 12,413 vértices de los ríos Meta, Manacacías e Iteviare."
    }},
    4: {{
      tag: "CONSTELACIÓN MULTIMODAL",
      titulo: "Diapositiva 5: Grafo Mixto Multimodal G=(V,E) (25 Vértices · 41 Aristas)",
      sub: "Flujo integral Pesca (P) → Muelle (M) → Acopio (A) → Frío (F) → Venta (V), evitando hábitats H y modelando deterioro D."
    }}
  }};

  document.getElementById("agy-slide-tag").textContent = metaSlides[agySlideActual].tag;
  document.getElementById("agy-slide-title").textContent = metaSlides[agySlideActual].titulo;
  document.getElementById("agy-slide-subtitle").textContent = metaSlides[agySlideActual].sub;

  document.getElementById("agy-gis-layers-panel").classList.toggle("visible", agySlideActual === 1);
  document.getElementById("agy-layout-switcher").style.display = (agySlideActual === 1) ? "none" : "inline-flex";
  const btnFolium = document.getElementById("agy-btn-folium-native");
  if (btnFolium) {{
    btnFolium.style.display = (!AGY_STANDALONE && agySlideActual === 1) ? "block" : "none";
  }}

  if (agySlideActual === 1) {{
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
  if (!sel) return;
  sel.innerHTML = '<option value="">Inspeccionar Vértice / Arista...</option>';

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

function agyRenderizarEscenaCompleta(inmediato = true) {{
  if (!agyEsSlideInteractivo(agySlideActual)) return;
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
    ${{agySlideActual === 1 ? `
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{(AGY_DATA.puntos_campo || []).length}}</div>
      <div class="agy-kpi-lbl">Puntos Campo</div>
    </div>` : ""}}
    ${{agySlideActual === 3 ? `
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{AGY_DATA.stats_rio.vertices.toLocaleString()}}</div>
      <div class="agy-kpi-lbl">Vértices rios.shp</div>
    </div>` : ""}}
    ${{agySlideActual === 4 ? `
    <div class="agy-kpi-card">
      <div class="agy-kpi-val">${{multimodalCount}}</div>
      <div class="agy-kpi-lbl">Nodos Transbordo</div>
    </div>` : ""}}
  `;

  // 2. Capas de Fondo: Retícula WGS84 (Slide 1: Mapa SIG) o Guías Dala (Slides 2..4)
  if (agySlideActual === 1) {{
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
        line.setAttribute("stroke", "rgba(255,237,215,0.06)");
        line.setAttribute("stroke-width", "0.5");
        line.setAttribute("stroke-dasharray", "2,2");
        layerGuides.appendChild(line);
      }}
      for (let lat = 3.5; lat <= 4.8; lat += 0.25) {{
        const pL = agyProyectarSIG(-74.20, lat, W, H);
        const pR = agyProyectarSIG(-71.02, lat, W, H);
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", pL.x);
        line.setAttribute("y1", pL.y);
        line.setAttribute("x2", pR.x);
        line.setAttribute("y2", pR.y);
        line.setAttribute("stroke", "rgba(255,237,215,0.06)");
        line.setAttribute("stroke-width", "0.5");
        line.setAttribute("stroke-dasharray", "2,2");
        layerGuides.appendChild(line);
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
        poly.setAttribute("stroke", "#2997ff");
        poly.setAttribute("stroke-width", "0.65");
        poly.setAttribute("stroke-opacity", "0.48");
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
        dot.setAttribute("fill", AGY_COLORES_NODO[pt.categoria] || "#9a9a9a");
        dot.setAttribute("fill-opacity", "0.82");
        dot.setAttribute("stroke", "#000000");
        dot.setAttribute("stroke-width", "0.4");
        dot.style.cursor = "pointer";
        dot.addEventListener("click", (ev) => {{
          ev.stopPropagation();
          agyInspeccionarPuntoCampo(pt);
        }});
        layerField.appendChild(dot);
      }});
    }}
  }} else if (agyLayoutModo === "hier") {{
    const padX = 84;
    AGY_ETAPAS_JERARQUICAS.forEach(et => {{
      const x = padX + (et.col / 5) * (W - 2 * padX);
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", x);
      line.setAttribute("y1", 92);
      line.setAttribute("x2", x);
      line.setAttribute("y2", H - 54);
      line.setAttribute("stroke", "rgba(255,237,215,0.08)");
      line.setAttribute("stroke-width", "1");
      line.setAttribute("stroke-dasharray", "4,4");
      layerGuides.appendChild(line);

      const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
      lbl.setAttribute("x", x);
      lbl.setAttribute("y", 82);
      lbl.setAttribute("text-anchor", "middle");
      lbl.setAttribute("fill", "#ffedd7");
      lbl.setAttribute("fill-opacity", "0.75");
      lbl.setAttribute("font-size", "12");
      lbl.setAttribute("font-family", "var(--font-mono)");
      lbl.textContent = et.titulo;
      layerGuides.appendChild(lbl);
    }});
  }} else {{
    const padX = 84, padYTop = 114;
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
      rect.setAttribute("rx", "20");
      rect.setAttribute("fill", idx === 1 ? "rgba(128, 82, 255, 0.03)" : "rgba(255, 255, 255, 0.01)");
      rect.setAttribute("stroke", idx === 1 ? "rgba(128, 82, 255, 0.24)" : "rgba(255, 237, 215, 0.08)");
      rect.setAttribute("stroke-width", "1");
      rect.setAttribute("stroke-dasharray", "4,4");
      layerGuides.appendChild(rect);

      const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
      lbl.setAttribute("x", zx + zw / 2);
      lbl.setAttribute("y", 82);
      lbl.setAttribute("text-anchor", "middle");
      lbl.setAttribute("fill", idx === 1 ? "#ffb829" : "#9a9a9a");
      lbl.setAttribute("font-size", "12");
      lbl.setAttribute("font-family", "var(--font-mono)");
      lbl.textContent = z.titulo;
      layerGuides.appendChild(lbl);
    }});

    if (agySlideActual === 3 || agySlideActual === 4) {{
      (AGY_DATA.rios_lineas || []).forEach(coords => {{
        const pts = coords.map(c => {{
          const p = agyProyectarGeoLupa(c[0], c[1], W, H, padX, padYTop);
          return `${{p.x.toFixed(1)}},${{p.y.toFixed(1)}}`;
        }}).join(" ");
        const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        poly.setAttribute("points", pts);
        poly.setAttribute("fill", "none");
        poly.setAttribute("stroke", "#2997ff");
        poly.setAttribute("stroke-width", agySlideActual === 3 ? "1.4" : "0.9");
        poly.setAttribute("stroke-opacity", agySlideActual === 3 ? "0.28" : "0.14");
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
    if (agySlideActual === 1) {{
      if (a.tipo_via === "fluvial" && !showNavRiversGIS) return;
      if (a.tipo_via !== "fluvial" && !showRoadsGIS) return;
    }}

    const p1 = agyPosiciones[a.origen];
    const p2 = agyPosiciones[a.destino];
    if (!p1 || !p2) return;

    const esSeleccionada = agySeleccion && agySeleccion.tipo === "arista" && agySeleccion.id === a.id;
    const esDeItinerario = aristasItinerario.has(a.id);
    const esVecinaHover = agyHoverNodeId && (a.origen === agyHoverNodeId || a.destino === agyHoverNodeId);

    let stroke = "#15846e";
    let marker = "url(#arrow-terrestre)";
    let dash = "";
    if (a.tipo_via === "fluvial") {{
      stroke = "#2997ff";
      marker = "url(#arrow-fluvial)";
      dash = agySlideActual === 1 ? "2,1.5" : "6,4";
    }} else if (a.tipo_via === "acceso_local") {{
      stroke = "#9a9a9a";
      marker = "url(#arrow-acceso)";
      dash = a.estado_via === "regular" ? (agySlideActual === 1 ? "1.5,1.5" : "4,4") : "";
    }} else {{
      if (a.estado_via === "mala") {{
        stroke = "#dc5000";
        marker = "url(#arrow-mala)";
      }} else if (a.estado_via === "regular") {{
        stroke = "#ffb829";
        marker = "url(#arrow-regular)";
      }}
    }}

    if (esSeleccionada || esDeItinerario) {{
      stroke = "#ffb829";
      marker = "url(#arrow-active)";
    }}

    let opacidad = "0.86";
    if (agyItinerarioActivo && !esDeItinerario && !esSeleccionada) {{
      opacidad = "0.16";
    }} else if (agyHoverNodeId && !esVecinaHover && !esSeleccionada) {{
      opacidad = "0.18";
    }} else if (esVecinaHover || esSeleccionada || esDeItinerario) {{
      opacidad = "1";
    }}

    let pathD = "";
    let lx = 0, ly = 0;

    if (agySlideActual === 1 && a.geometria_muestra && a.geometria_muestra.length >= 2) {{
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

    const baseWidth = agySlideActual === 1
      ? ((esSeleccionada || esDeItinerario) ? "1.6" : "0.85")
      : ((esSeleccionada || esDeItinerario) ? "3.8" : (esVecinaHover ? "2.8" : "2.0"));
    pathVis.setAttribute("stroke-width", baseWidth);
    pathVis.setAttribute("stroke-opacity", opacidad);
    if (dash || esDeItinerario) {{
      pathVis.setAttribute("stroke-dasharray", esDeItinerario ? (agySlideActual === 1 ? "3,2" : "10,6") : dash);
    }}
    if (agySlideActual !== 1) {{
      pathVis.setAttribute("marker-end", marker);
    }}

    const pathHit = document.createElementNS("http://www.w3.org/2000/svg", "path");
    pathHit.setAttribute("d", pathD);
    pathHit.setAttribute("class", "agy-edge-hit");
    if (agySlideActual === 1) pathHit.setAttribute("stroke-width", "5px");
    pathHit.addEventListener("click", (ev) => {{
      ev.stopPropagation();
      agyInspeccionarArista(a);
    }});

    gEdge.appendChild(pathVis);
    gEdge.appendChild(pathHit);

    let txt = null;
    if (agySlideActual !== 1) {{
      txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
      txt.setAttribute("x", lx.toFixed(1));
      txt.setAttribute("y", ly.toFixed(1));
      txt.setAttribute("text-anchor", "middle");
      txt.setAttribute("fill", (esSeleccionada || esDeItinerario) ? "#ffb829" : "#9a9a9a");
      txt.setAttribute("fill-opacity", opacidad);
      txt.setAttribute("font-family", "var(--font-mono)");
      txt.setAttribute("font-size", "10");
      txt.setAttribute("font-weight", (esSeleccionada || esDeItinerario) ? "700" : "500");
      txt.setAttribute("style", "paint-order: stroke; stroke: #000000; stroke-width: 3.5px; stroke-linejoin: round;");
      txt.setAttribute("pointer-events", "none");
      txt.textContent = `${{a.distancia_km}}km·${{a.tiempo_efectivo_h.toFixed(2)}}h`;
      gEdge.appendChild(txt);
    }}

    layerEdges.appendChild(gEdge);
    agyDomEdges[a.id] = {{ arista: a, pathVis, pathHit, txt }};
  }});

  // 5. Dibujar Nodos (Vértices con estética de Constelación Dala)
  const showNodesGIS = document.getElementById("chk-layer-nodes")?.checked ?? true;
  if (agySlideActual === 1 && !showNodesGIS) return;

  nodos.forEach(n => {{
    const pos = agyPosiciones[n.id];
    if (!pos) return;

    const esSeleccionado = agySeleccion && agySeleccion.tipo === "nodo" && agySeleccion.id === n.id;
    const esDeItin = nodosItinerario.has(n.id);
    const color = AGY_COLORES_NODO[n.tipo] || "#8052ff";
    const esGIS = (agySlideActual === 1);

    const gNode = document.createElementNS("http://www.w3.org/2000/svg", "g");
    gNode.setAttribute("class", "agy-node-group");
    gNode.setAttribute("transform", `translate(${{pos.x.toFixed(1)}}, ${{pos.y.toFixed(1)}})`);

    // Halo de constelación Dala + satélites triangulares cuando está seleccionado o es nodo crítico
    if (!esGIS && (n.tipo === "D" || n.tipo === "H" || esSeleccionado || esDeItin || (agySlideActual === 4 && n.es_multimodal))) {{
      const halo = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      halo.setAttribute("r", esSeleccionado ? "24" : "19");
      halo.setAttribute("fill", esSeleccionado ? "rgba(128,82,255,0.18)" : (n.tipo === "D" ? "rgba(220,80,0,0.15)" : (n.tipo === "H" ? "rgba(16,185,129,0.15)" : "rgba(255,184,41,0.12)")));
      halo.setAttribute("stroke", esSeleccionado ? "#ffb829" : color);
      halo.setAttribute("stroke-width", "1.4");
      if (n.tipo === "H" || (agySlideActual === 4 && n.es_multimodal && !esSeleccionado)) {{
        halo.setAttribute("stroke-dasharray", "3,3");
      }}
      gNode.appendChild(halo);

      // Partículas triangulares orbitales Dala alrededor del nodo activo
      if (esSeleccionado || esDeItin) {{
        [0, 120, 240].forEach((deg, idx) => {{
          const rad = (deg * Math.PI) / 180;
          const tx = Math.cos(rad) * 24;
          const ty = Math.sin(rad) * 24;
          const tri = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
          tri.setAttribute("points", `${{tx}},${{ty - 3}} ${{tx + 2.6}},${{ty + 1.6}} ${{tx - 2.6}},${{ty + 1.6}}`);
          tri.setAttribute("fill", idx === 0 ? "#ffb829" : "#8052ff");
          gNode.appendChild(tri);
        }});
      }}
    }}

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("class", "agy-node-circle");
    circle.setAttribute("r", esGIS ? (esSeleccionado ? "3.4" : "2.2") : "14");
    circle.setAttribute("fill", esGIS ? color : "#000000");
    circle.setAttribute("stroke", esSeleccionado ? "#ffb829" : (esGIS ? "#000000" : color));
    circle.setAttribute("stroke-width", esGIS ? "0.6" : (esSeleccionado ? "3.2" : "2.4"));
    gNode.appendChild(circle);

    const tooltip = document.createElementNS("http://www.w3.org/2000/svg", "title");
    tooltip.textContent = `[${{n.id}}] ${{n.nombre}} (${{n.tipo_descripcion}}) — ${{n.latitud.toFixed(4)}}°N, ${{n.longitud.toFixed(4)}}°W`;
    gNode.appendChild(tooltip);

    if (!esGIS) {{
      const lblId = document.createElementNS("http://www.w3.org/2000/svg", "text");
      lblId.setAttribute("text-anchor", "middle");
      lblId.setAttribute("dy", "3.6");
      lblId.setAttribute("fill", esSeleccionado ? "#ffb829" : "#ffffff");
      lblId.setAttribute("font-family", "var(--font-mono)");
      lblId.setAttribute("font-size", "10");
      lblId.setAttribute("font-weight", "700");
      lblId.setAttribute("pointer-events", "none");
      lblId.textContent = n.id;
      gNode.appendChild(lblId);
    }}

    const lblName = document.createElementNS("http://www.w3.org/2000/svg", "text");
    lblName.setAttribute("text-anchor", "middle");
    lblName.setAttribute("dy", esGIS ? "-3.4" : "27");
    lblName.setAttribute("fill", esSeleccionado ? "#ffb829" : "#ffedd7");
    lblName.setAttribute("font-family", esGIS ? "var(--font-mono)" : "var(--font-sans)");
    lblName.setAttribute("font-size", esGIS ? "3" : "11");
    lblName.setAttribute("font-weight", esGIS ? "700" : "500");
    lblName.setAttribute("style", esGIS
      ? "paint-order: stroke; stroke: #000000; stroke-width: 0.8px; stroke-linejoin: round;"
      : "paint-order: stroke; stroke: #000000; stroke-width: 4px; stroke-linejoin: round;");
    lblName.setAttribute("pointer-events", "none");
    lblName.textContent = esGIS ? n.id : agyNombreCompacto(n.nombre);
    gNode.appendChild(lblName);

    gNode.addEventListener("mouseenter", () => {{
      if (!agyDraggingNode && agySlideActual !== 1) {{
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
      if (agySlideActual === 1) return;
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

function agyActualizarOpacidadHover() {{
  Object.keys(agyDomEdges).forEach(edgeId => {{
    const ref = agyDomEdges[edgeId];
    const a = ref.arista;
    const esSeleccionada = agySeleccion && agySeleccion.tipo === "arista" && agySeleccion.id === a.id;
    const esVecinaHover = agyHoverNodeId && (a.origen === agyHoverNodeId || a.destino === agyHoverNodeId);
    let op = "0.86";
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

function agyMostrarResumenSubgrafo() {{
  agySeleccion = null;
  const {{ nodos, aristas }} = agySubgrafoActual();
  document.getElementById("agy-insp-title").textContent = "Estructura del Grafo en Memoria";
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `${{nodos.length}} V · ${{aristas.length}} E`;
  badge.style.color = "var(--color-saffron-spark)";

  const adjLines = nodos.map(n => {{
    const salientes = aristas.filter(a => a.origen === n.id).map(a => `${{a.destino}}(w=${{a.peso_combinado.toFixed(3)}})`).join(", ");
    return `_adyacencia["${{n.id}}"] -> [${{salientes}}]`;
  }}).join("\\n");

  const chipsNodos = nodos.map(n =>
    `<span class="agy-chip" onclick="agySeleccionarElementoRapido('N:${{n.id}}')"><span class="agy-leg-dot" style="background:${{AGY_COLORES_NODO[n.tipo]}}"></span>${{n.id}} · ${{agyNombreCompacto(n.nombre)}}</span>`
  ).join("");

  const chipsAristas = aristas.slice(0, 14).map(a =>
    `<span class="agy-chip" onclick="agySeleccionarElementoRapido('E:${{a.id}}')">${{a.origen}} → ${{a.destino}} (${{a.tipo_via}})</span>`
  ).join("");

  const flp = AGY_DATA.flp || {{}};

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <strong>Exploración de Constelación G=(V,E):</strong> Haz clic sobre cualquier vértice o arista en el lienzo negro para inspeccionar sus atributos reales, grados topológicos y desglose multicriterio.
    </div>

    <div class="agy-prop-group">
      <h4><span>Parámetros del Subgrafo Activo</span><span>G=(V,E)</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Vértices activos |V|</span><span class="agy-prop-val">${{nodos.length}} nodos</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Aristas dirigidas |E|</span><span class="agy-prop-val">${{aristas.length}} arcos</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Estructura base</span><span class="agy-prop-val">Lista Adyacencia O(|V|+|E|)</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Localización FLP (${{flp.metodo || 'greedy'}})</span><span class="agy-prop-val">Centro óptimo: ${{(flp.centros_abiertos || ['F02']).join(', ')}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Límite térmico FAO</span><span class="agy-prop-val">${{AGY_DATA.constantes.TIEMPO_MAX_SIN_FRIO_H}} h sin frío</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>Vértices (${{nodos.length}})</span><span>Clic para inspeccionar</span></h4>
      <div class="agy-chip-list">${{chipsNodos}}</div>
    </div>

    <div class="agy-prop-group">
      <h4><span>Conexiones Dirigidas</span><span>Muestra activa</span></h4>
      <div class="agy-chip-list">${{chipsAristas}}</div>
    </div>

    <div class="agy-prop-group">
      <h4><span>_adyacencia en Memoria</span><span>Python dict</span></h4>
      <div class="agy-code-block">${{adjLines}}</div>
    </div>
  `;
}}

function agyMostrarComparadorEscenarios() {{
  const lista = agyDestinoModo === "PG" ? (AGY_DATA.escenarios_pg || []) : (AGY_DATA.escenarios_bog || []);
  const flp = AGY_DATA.flp || {{}};
  document.getElementById("agy-insp-title").textContent = `Evaluación de Escenarios (${{agyDestinoModo === "PG" ? "Pto. Gaitán" : "Bogotá"}})`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `${{lista.length}} Escenarios`;
  badge.style.color = "var(--color-saffron-spark)";

  const filas = lista.map((e, i) => `
    <tr class="${{i === 2 ? 'row-optimal' : ''}}">
      <td>${{e.nombre}}</td>
      <td style="text-align:right">${{e.num_centros}}</td>
      <td style="text-align:right">${{e.pct_perdida.toFixed(1)}}%</td>
      <td style="text-align:right">$${{(e.costo_total_cop / 1e6).toFixed(2)}}M</td>
      <td style="text-align:right">$${{e.costo_por_kg_cop.toLocaleString()}}</td>
    </tr>
  `).join("");

  const asigs = Object.entries(flp.asignaciones || {{}}).map(([p, f]) =>
    `<span class="agy-chip" onclick="agySeleccionarElementoRapido('N:${{p}}')">${{p}} → ${{f}}</span>`
  ).join("");

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <strong>Facility Location Problem (FLP Greedy):</strong><br>
      Centro abierto óptimo: <strong>${{(flp.centros_abiertos || ['F02']).join(", ")}}</strong> · Tiempo máx. acceso: <strong>${{flp.tiempo_max_acceso_h}} h</strong> · Cobertura: <strong>100%</strong>.
    </div>

    <div class="agy-prop-group">
      <h4><span>Comparativa Real de Escenarios</span><span>Destino: ${{agyDestinoModo}}</span></h4>
      <table class="agy-table">
        <thead>
          <tr>
            <th>Escenario</th>
            <th style="text-align:right">CF</th>
            <th style="text-align:right">Pérd.</th>
            <th style="text-align:right">Costo</th>
            <th style="text-align:right">$/kg</th>
          </tr>
        </thead>
        <tbody>${{filas}}</tbody>
      </table>
    </div>

    <div class="agy-prop-group">
      <h4><span>Asignaciones Pesca → Frío</span><span>7 orígenes</span></h4>
      <div class="agy-chip-list">${{asigs}}</div>
    </div>
  `;
}}

function agyInspeccionarNodo(n) {{
  agySeleccion = {{ tipo: "nodo", id: n.id }};
  agyRenderizarEscenaCompleta(true);

  document.getElementById("agy-insp-title").textContent = `Vértice ${{n.id}}: ${{n.nombre}}`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `Tipo ${{n.tipo}} · ${{n.tipo_descripcion}}`;
  badge.style.color = AGY_COLORES_NODO[n.tipo] || "#8052ff";

  const aristasSalientes = AGY_DATA.aristas.filter(a => a.origen === n.id);
  const aristasEntrantes = AGY_DATA.aristas.filter(a => a.destino === n.id);

  const chipsSalientes = aristasSalientes.length
    ? aristasSalientes.map(a => `<span class="agy-chip" onclick="agySeleccionarElementoRapido('E:${{a.id}}')">→ ${{a.destino}} (${{a.tipo_via}}, ${{a.distancia_km}} km, w=${{a.peso_combinado.toFixed(3)}})</span>`).join("")
    : `<span style="color:var(--color-ash-gray);font-size:12px;">Ninguna (grado de salida = 0)</span>`;

  const chipsEntrantes = aristasEntrantes.length
    ? aristasEntrantes.map(a => `<span class="agy-chip" onclick="agySeleccionarElementoRapido('E:${{a.id}}')">← ${{a.origen}} (${{a.tipo_via}}, ${{a.distancia_km}} km)</span>`).join("")
    : `<span style="color:var(--color-ash-gray);font-size:12px;">Ninguna (grado de entrada = 0)</span>`;

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
      <strong>Efecto algorítmico en el modelo:</strong><br>
      ${{n.efecto_vertice}}
    </div>

    <div class="agy-prop-group">
      <h4><span>1. Identidad y Coordenadas WGS84</span><span>Nodo</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Identificador</span><span class="agy-prop-val">${{n.id}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Nombre</span><span class="agy-prop-val">${{n.nombre}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Rol logístico</span><span class="agy-prop-val">${{n.tipo}} · ${{n.tipo_descripcion}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Coordenadas GPS</span><span class="agy-prop-val">${{n.latitud.toFixed(4)}}°N, ${{n.longitud.toFixed(4)}}°W</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Modos conectados</span><span class="agy-prop-val">${{(n.modos_conectados || []).join(" + ") || "—"}}</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>2. Capacidades, Producción y Costos</span><span>Métricas</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Producción diaria</span><span class="agy-prop-val">${{n.produccion_kg_dia.toFixed(2)}} kg/día</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Capacidad frigorífica</span><span class="agy-prop-val">${{n.capacidad_frio_kg.toLocaleString()}} kg</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Costo fijo mensual</span><span class="agy-prop-val">$${{n.costo_fijo.toLocaleString()}} COP</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Demanda comercial</span><span class="agy-prop-val">${{n.metadata.demanda_kg_dia || 0}} kg/día</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Cuenca hidrográfica</span><span class="agy-prop-val">${{n.metadata.rio || "—"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Pescadores censados</span><span class="agy-prop-val">${{n.metadata.pescadores || "—"}}</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>3. Infraestructura y Restricciones</span><span>Estado</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Energía eléctrica</span><span class="agy-prop-val">${{n.tiene_energia ? "Sí (Continua)" : "No disponible"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Agua potable</span><span class="agy-prop-val">${{n.tiene_agua ? "Sí" : "No"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Infraestructura muelle</span><span class="agy-prop-val">${{n.tiene_muelle ? "Sí" : "No"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Restricción ambiental</span><span class="agy-prop-val">${{n.restriccion_ambiental ? "Hábitat protegido (H)" : "Sin restricción"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Elegible FLP</span><span class="agy-prop-val">${{n.es_candidato_frio ? "Candidato válido" : "No elegible"}}</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>4. Conectividad en Lista de Adyacencia</span><span>deg⁻ / deg⁺</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Grado de Entrada (deg⁻)</span><span class="agy-prop-val">${{n.grado_entrada}} aristas entrantes</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Grado de Salida (deg⁺)</span><span class="agy-prop-val">${{n.grado_salida}} aristas salientes</span></div>
      <div style="margin-top:8px;color:var(--color-warm-cream);font-size:12px;">Aristas salientes desde ${{n.id}}:</div>
      <div class="agy-chip-list">${{chipsSalientes}}</div>
      <div style="margin-top:8px;color:var(--color-warm-cream);font-size:12px;">Aristas entrantes hacia ${{n.id}}:</div>
      <div class="agy-chip-list">${{chipsEntrantes}}</div>
    </div>

    <div class="agy-prop-group">
      <h4><span>5. Instancia Serializada</span><span>src/models/nodo.py</span></h4>
      <div class="agy-code-block">${{pythonObj}}</div>
    </div>
  `;
}}

function agyInspeccionarArista(a) {{
  agySeleccion = {{ tipo: "arista", id: a.id }};
  agyRenderizarEscenaCompleta(true);

  const nO = AGY_DATA.nodos.find(x => x.id === a.origen);
  const nD = AGY_DATA.nodos.find(x => x.id === a.destino);

  document.getElementById("agy-insp-title").textContent = `Arista #${{String(a.idx).padStart(2, '0')}}: ${{a.origen}} → ${{a.destino}}`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `${{a.tipo_via}} (${{a.estado_via}})`;
  badge.style.color = a.tipo_via === "fluvial" ? "#2997ff" : (a.tipo_via === "acceso_local" ? "#9a9a9a" : "#15846e");

  const w = a.peso_combinado || 0.0001;
  const d = a.desglose_peso;
  const pct = (val) => Math.min(100, Math.round((val / w) * 100));

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <strong>${{nO ? nO.nombre : a.origen}} (${{a.origen}}) → ${{nD ? nD.nombre : a.destino}} (${{a.destino}})</strong><br>
      Geometría real almacenada: <strong>${{a.num_puntos_geometria}} vértices GPS</strong> (${{a.tipo_via === "fluvial" ? "Cauce hidrográfico rios.shp" : "Trazado vial OSRM"}}).
      <div class="agy-chip-list" style="margin-top:8px;">
        <span class="agy-chip" onclick="agySeleccionarElementoRapido('N:${{a.origen}}')">Nodo origen ${{a.origen}}</span>
        <span class="agy-chip" onclick="agySeleccionarElementoRapido('N:${{a.destino}}')">Nodo destino ${{a.destino}}</span>
      </div>
    </div>

    <div class="agy-prop-group">
      <h4><span>Peso Multicriterio</span><span>w_ij = ${{a.peso_combinado.toFixed(4)}}</span></h4>
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
      <div class="agy-prop-row"><span class="agy-prop-key">Distancia del tramo</span><span class="agy-prop-val">${{a.distancia_km.toFixed(2)}} km</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Tiempo base</span><span class="agy-prop-val">${{a.tiempo_h.toFixed(2)}} h</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Flujo vehicular</span><span class="agy-prop-val">${{a.flujo_vehicular}} (×${{a.factor_flujo}})</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Tiempo de espera</span><span class="agy-prop-val">${{a.tiempo_espera_h.toFixed(2)}} h</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Tiempo efectivo</span><span class="agy-prop-val">${{a.tiempo_efectivo_h.toFixed(3)}} h</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Velocidad efectiva</span><span class="agy-prop-val">${{a.velocidad_efectiva_kmh.toFixed(1)}} km/h</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>2. Costos Logísticos y Peajes</span><span>COP</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Flete base</span><span class="agy-prop-val">$${{a.costo_cop.toLocaleString()}} COP</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Casetas de peaje</span><span class="agy-prop-val">${{a.peajes}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Costo de peajes</span><span class="agy-prop-val">$${{a.costo_peajes_cop.toLocaleString()}} COP</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Costo total</span><span class="agy-prop-val">$${{a.costo_total_cop.toLocaleString()}} COP</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>3. Estado Vial, Riesgo y Deterioro</span><span>22 Variables</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Modalidad de vía</span><span class="agy-prop-val">${{a.tipo_via}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Estado superficial</span><span class="agy-prop-val">${{a.estado_via}} (×${{a.factor_estado_via}})</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Confiabilidad vial</span><span class="agy-prop-val">${{a.confiabilidad_via.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Probabilidad de retraso</span><span class="agy-prop-val">${{a.prob_retraso.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Riesgo térmico</span><span class="agy-prop-val">${{a.riesgo_termico.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Riesgo de accidente</span><span class="agy-prop-val">${{a.riesgo_accidente.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Riesgo de derrumbe</span><span class="agy-prop-val">${{a.riesgo_derrumbe.toFixed(2)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Riesgo combinado</span><span class="agy-prop-val">${{a.riesgo_combinado.toFixed(4)}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Pérdida esperada (L_ij)</span><span class="agy-prop-val">${{a.perdida_esperada.toFixed(2)}}%</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Capacidad / Peso máx.</span><span class="agy-prop-val">${{a.capacidad_kg}} kg · ${{a.restriccion_peso_ton}} t</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Restricción horaria</span><span class="agy-prop-val">${{a.horario_restriccion}}</span></div>
    </div>
  `;
}}

function agyInspeccionarPuntoCampo(pt) {{
  agySeleccion = {{ tipo: "campo", id: pt.id }};
  document.getElementById("agy-insp-title").textContent = `Punto de Campo ${{pt.id}}`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = pt.categoria_texto;
  badge.style.color = AGY_COLORES_NODO[pt.categoria] || "#2997ff";

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <strong>Registro de Levantamiento Georreferenciado (puntos_campo.geojson)</strong><br>
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
  document.getElementById("agy-insp-title").textContent = `Itinerario ${{it.origen_id}} → ${{it.destino_id}}`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `${{it.distancia_total_km}} km · ${{it.tiempo_total_h}} h`;
  badge.style.color = "var(--color-saffron-spark)";

  const filasTramos = (it.tramos || []).map((t, i) => `
    <div class="agy-prop-row" style="cursor:pointer" onclick="agySeleccionarElementoRapido('E:${{t.origen}}->${{t.destino}}')">
      <span class="agy-prop-key">${{i+1}}. ${{t.origen}} → ${{t.destino}} (${{t.tipo_via}})</span>
      <span class="agy-prop-val">${{t.distancia_km}} km · ${{t.tiempo_h}} h ${{t.efecto ? '· ' + t.efecto : ''}}</span>
    </div>
  `).join("");

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <strong>Secuencia Multimodal Óptima (ADR-06):</strong><br>
      ${{it.ruta.join(" → ")}}<br>
      ${{it.nodos_restringidos_evitados.length ? `<strong>Hábitats protegidos evitados:</strong> ${{it.nodos_restringidos_evitados.join(", ")}}` : "Ruta libre de restricciones ambientales."}}
    </div>
    <div class="agy-prop-group">
      <h4><span>Métricas Consolidadas</span><span>Dijkstra + ColdChain</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Acopio obligatorio</span><span class="agy-prop-val">${{it.acopio_id || "Directo urbano (P06)"}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Centro de Frío</span><span class="agy-prop-val">${{it.centro_frio_id}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Distancia Total</span><span class="agy-prop-val">${{it.distancia_total_km}} km</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Tiempo Total</span><span class="agy-prop-val">${{it.tiempo_total_h}} h</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Carga Inicial → Final</span><span class="agy-prop-val">${{it.kg_inicial}} kg → ${{it.kg_final}} kg</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Pérdida Térmica</span><span class="agy-prop-val">${{it.perdida_pct}}%</span></div>
    </div>
    <div class="agy-prop-group">
      <h4><span>Tramos del Recorrido</span><span>Clic para inspeccionar</span></h4>
      ${{filasTramos}}
    </div>
  `;
}}

// Inicialización general de eventos y vistas
(function initEventosPlataforma() {{
  if (window.location.search.includes("folium_only=1")) {{
    document.body.classList.add("folium-embed-only");
    return;
  }}
  agyPoblarSelectorItinerarios();
  agyInicializarSlidesEditoriales();
  agyIniciarConstelacionDala();

  const svg = document.getElementById("agy-graph-svg");

  svg.addEventListener("mousedown", (ev) => {{
    if (agyDraggingNode) return;
    agyPanning = true;
    agyPanStart = {{ x: ev.clientX - agyTransform.x, y: ev.clientY - agyTransform.y }};
  }});

  window.addEventListener("mousemove", (ev) => {{
    if (!agyEsSlideInteractivo(agySlideActual)) return;
    const rect = svg.getBoundingClientRect();
    const sx = (ev.clientX - rect.left - agyTransform.x) / agyTransform.k;
    const sy = (ev.clientY - rect.top - agyTransform.y) / agyTransform.k;

    if (agySlideActual === 1) {{
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
    if (!agyEsSlideInteractivo(agySlideActual)) return;
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

  window.addEventListener("keydown", (ev) => {{
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (ev.key === "ArrowRight") {{
      agyCambiarSlide(agySlideActual + 1, true);
    }} else if (ev.key === "ArrowLeft") {{
      agyCambiarSlide(agySlideActual - 1, true);
    }} else if (["1", "2", "3", "4", "5", "6", "7", "8"].includes(ev.key)) {{
      agyCambiarSlide(Number(ev.key) - 1, true);
    }} else if (ev.key === "r" || ev.key === "R") {{
      agyResetVista();
    }} else if (ev.key === "Escape") {{
      agyItinerarioActivo = "";
      const selIt = document.getElementById("agy-itin-select");
      if (selIt) selIt.value = "";
      agyRenderizarEscenaCompleta(true);
      agyMostrarResumenSubgrafo();
    }}
  }});

  window.addEventListener("resize", () => {{
    if (agyEsSlideInteractivo(agySlideActual)) {{
      agyRenderizarEscenaCompleta(true);
    }}
  }});

  setTimeout(() => {{
    agyCambiarSlide(0, true);
  }}, 20);
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
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%23000000'/><polygon points='16,5 27,25 5,25' fill='none' stroke='%238052ff' stroke-width='2.5'/><circle cx='16' cy='18' r='3' fill='%23ffb829'/></svg>">
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #000000;
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
