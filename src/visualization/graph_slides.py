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
from ..optimization.cold_chain import SimuladorCadenaFrio
from ..optimization.scenario_evaluator import EvaluadorEscenarios
from ..optimization.itinerarios import calcular_itinerarios
from ..optimization.eca_solutions import ejecutar_suite_eca
from ..utils.constants import (
    ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA,
    TIEMPO_MAX_SIN_FRIO_H, TIEMPO_TRANSBORDO_MUELLE_H, FACTOR_DETERIORO_ZONA_D,
    COSTO_FIJO_CENTRO_FRIO, TEMP_AMBIENTE_LLANOS_C, VIDA_UTIL_HORAS, COSTO_PERDIDA_KG,
)
from ..utils.data_loader import DATA_DIR, OUTPUT_DIR

GEOJSON_DIR = DATA_DIR / "geojson"


def _cargar_puntos_campo_reales() -> List[Dict[str, Any]]:
    """
    Carga los 49 puntos de campo reales desde `data/geojson/puntos_campo.geojson`
    enriquecidos con su corredor fluvial real sobre `rios.shp`, condiciones de
    cuesta del río (aguas abajo vs. cuesta arriba), velocidades en lancha y frío.
    """
    try:
        from .river_network import calcular_corredor_fluvial_puntos_campo
        corredor = calcular_corredor_fluvial_puntos_campo()
        pts = corredor.get("puntos", [])
        if pts:
            return list(pts)
    except Exception:
        pass

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
                "kg_perdido": round(m.kg_perdido, 2),
                "kg_total": round(m.kg_total_transportado, 2),
                "costo_transporte_cop": round(m.costo_transporte_cop, 0),
                "costo_centros_cop": round(m.costo_centros_cop, 0),
                "costo_perdida_cop": round(m.costo_perdida_cop, 0),
                "costo_total_cop": round(m.costo_total_cop, 0),
                "costo_por_kg_cop": round(m.costo_por_kg_cop, 0),
                "cobertura_pesca_pct": round(m.cobertura_pesca_pct, 1),
                "tiempo_max_sin_frio_h": round(m.tiempo_max_sin_frio_h, 2),
                "cumplimiento_termico_pct": round(m.cumplimiento_termico_pct, 1),
            })
        for m in ev.comparar_escenarios(defs_esc, "V_BOG1"):
            escenarios_bog.append({
                "nombre": m.nombre,
                "num_centros": m.num_centros,
                "centros_ids": m.centros_ids,
                "pct_perdida": round(m.pct_perdida, 2),
                "kg_perdido": round(m.kg_perdido, 2),
                "kg_total": round(m.kg_total_transportado, 2),
                "costo_transporte_cop": round(m.costo_transporte_cop, 0),
                "costo_centros_cop": round(m.costo_centros_cop, 0),
                "costo_perdida_cop": round(m.costo_perdida_cop, 0),
                "costo_total_cop": round(m.costo_total_cop, 0),
                "costo_por_kg_cop": round(m.costo_por_kg_cop, 0),
                "cobertura_pesca_pct": round(m.cobertura_pesca_pct, 1),
                "tiempo_max_sin_frio_h": round(m.tiempo_max_sin_frio_h, 2),
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
    eca_data = ejecutar_suite_eca(grafo)

    # 9. Auditoría FLP Detallada (Paso 2 & Paso 3: Origen, Filtro, Matriz Dijkstra 7×5, P05 Térmica, F02→F03)
    flp_detallado: Dict[str, Any] = {}
    try:
        origenes_p = sorted(grafo.nodos_por_tipo("P"), key=lambda x: x.id)
        oferta_tot = round(sum(p.produccion_kg_dia for p in origenes_p), 2)
        orden_cands = ["F01", "F02", "A01", "A02", "F03"]
        cands_validos = [grafo.obtener_nodo(cid) for cid in orden_cands if grafo.obtener_nodo(cid)]
        cands_todos = list(cands_validos)
        n_a03 = grafo.obtener_nodo("A03")
        if n_a03:
            cands_todos.append(n_a03)

        provenance_coords = {
            "F01": {
                "origen": "Manual +0.005° (~780 m) desde P01 ASOPESGA",
                "commit": "9af49ef (nodos_centros.csv)",
                "regla": "Ubicación de análisis en zona norte de pesca (río Meta).",
            },
            "F02": {
                "origen": "Cabecera urbana Puerto Gaitán (constants.py · PUERTO_GAITAN_COORDS)",
                "commit": "9af49ef (nodos_centros.csv)",
                "regla": "Nodo articulador urbano con energía trifásica y salida directa a RN 40.",
            },
            "F03": {
                "origen": "Cabecera urbana Puerto López (constants.py · PUERTO_LOPEZ_COORDS)",
                "commit": "9af49ef (nodos_centros.csv)",
                "regla": "Nodo de relevo a 126 km al occidente sobre Ruta Nacional 40.",
            },
            "A01": {
                "origen": "Mismas coordenadas exactas que P03 San Miguel (4.2900, -72.0600)",
                "commit": "9af49ef (nodos_centros.csv)",
                "regla": "Acopio sur (cuenca Manacacías/Iteviare) con energía eléctrica continua.",
            },
            "A02": {
                "origen": "Punto medio aritmético exacto entre P02 (4.33, -72.10) y P07 (4.34, -72.11)",
                "commit": "9af49ef (nodos_centros.csv)",
                "regla": "Acopio norte intermedio para P01, P02, P06 y P07.",
            },
            "A03": {
                "origen": "Mismas coordenadas exactas que P05 Puerto Trujillo (4.1500, -71.5000)",
                "commit": "9af49ef (nodos_centros.csv)",
                "regla": "DESCARTADO del FLP porque tiene_energia = 0 (sin red eléctrica estable).",
            },
        }

        matriz_7x5 = []
        stats_cand: Dict[str, Dict[str, Any]] = {
            c.id: {"cubiertos": 0, "cubiertos_ids": [], "fallidos": [], "tiempos_ok": []}
            for c in cands_validos
        }
        for p in origenes_p:
            celdas = {}
            for c in cands_validos:
                dist_t, cam_t = dijkstra(grafo, p.id, c.id, criterio="tiempo_h", evitar_restringidos=True)
                if cam_t and dist_t < float("inf"):
                    t_h = round(dist_t, 2)
                    camino = list(cam_t)
                    tramos_c = []
                    dist_sum = 0.0
                    costo_sum = 0.0
                    for idx_c in range(len(camino) - 1):
                        ar_c = grafo.obtener_arista(camino[idx_c], camino[idx_c + 1])
                        if ar_c:
                            dist_sum += ar_c.distancia_km
                            costo_sum += ar_c.costo_total_cop
                            tramos_c.append({
                                "u": ar_c.origen,
                                "v": ar_c.destino,
                                "tipo_via": ar_c.tipo_via,
                                "tiempo_h": round(ar_c.tiempo_h, 2),
                                "distancia_km": round(ar_c.distancia_km, 2),
                                "costo_cop": round(ar_c.costo_total_cop, 0),
                                "estado_via": ar_c.estado_via,
                            })
                    cumple = t_h <= TIEMPO_MAX_SIN_FRIO_H
                    if cumple:
                        stats_cand[c.id]["cubiertos"] += 1
                        stats_cand[c.id]["cubiertos_ids"].append(p.id)
                        stats_cand[c.id]["tiempos_ok"].append(t_h)
                    else:
                        stats_cand[c.id]["fallidos"].append(f"{p.id} ({t_h:.2f} h > 4.0 h)")
                    nota = ""
                    if p.id == "P04" and c.id in ("A01", "F02", "F03"):
                        nota = "Desvío ecológico obligatorio: P04→H01→A01 (0.20 h) bloqueado por H01 (restriccion_ambiental=True); toma P04→A01 (0.25 h)."
                    elif not cumple:
                        nota = f"Supera el límite sanitario FAO ({t_h:.2f} h > {TIEMPO_MAX_SIN_FRIO_H:.1f} h): excluido de cubiertos_por_este[{c.id}]."
                    celdas[c.id] = {
                        "encontrado": True,
                        "tiempo_h": t_h,
                        "cumple_fao": cumple,
                        "camino": camino,
                        "dist_km": round(dist_sum, 2),
                        "costo_cop": round(costo_sum, 0),
                        "tramos": tramos_c,
                        "nota": nota,
                    }
                else:
                    stats_cand[c.id]["fallidos"].append(f"{p.id} (∞ sin ruta dirigida)")
                    celdas[c.id] = {
                        "encontrado": False,
                        "tiempo_h": None,
                        "cumple_fao": False,
                        "camino": [],
                        "dist_km": 0.0,
                        "costo_cop": 0.0,
                        "tramos": [],
                        "nota": f"Sin camino dirigido hacia {c.id} en el grafo G=(V,E).",
                    }
            matriz_7x5.append({
                "origen_id": p.id,
                "nombre": p.nombre,
                "produccion_kg_dia": round(p.produccion_kg_dia, 2),
                "pescadores": int(p.metadata.get("pescadores", 0)),
                "rio": str(p.metadata.get("rio", "")),
                "celdas": celdas,
            })

        candidatos_info = []
        for c in cands_todos:
            st = stats_cand.get(c.id, {"cubiertos": 0, "cubiertos_ids": [], "fallidos": [], "tiempos_ok": []})
            n_cub = st["cubiertos"]
            t_max = round(max(st["tiempos_ok"]), 2) if st["tiempos_ok"] else None
            prov = provenance_coords.get(c.id, {"origen": "nodos_centros.csv", "commit": "—", "regla": ""})
            candidatos_info.append({
                "id": c.id,
                "nombre": c.nombre,
                "tipo": c.tipo,
                "lat": round(c.latitud, 4),
                "lon": round(c.longitud, 4),
                "capacidad_frio_kg": int(c.capacidad_frio_kg),
                "costo_fijo_csv": int(c.costo_fijo),
                "costo_eval_esc": int(COSTO_FIJO_CENTRO_FRIO) if c.tipo == "F" else int(c.costo_fijo),
                "tiene_energia": bool(c.tiene_energia),
                "restriccion_ambiental": bool(c.restriccion_ambiental),
                "es_candidato": bool(c.es_candidato_frio),
                "cobertura_n": n_cub,
                "cubiertos_ids": list(st["cubiertos_ids"]),
                "fallidos": list(st["fallidos"]),
                "cobertura_pct": round(n_cub / max(len(origenes_p), 1) * 100.0, 1),
                "tiempo_max_h": t_max,
                "uso_cap_pct": round(oferta_tot / c.capacidad_frio_kg * 100.0, 1) if c.capacidad_frio_kg > 0 else 0.0,
                "origen_coords": prov["origen"],
                "origen_commit": prov["commit"],
                "regla": prov["regla"],
            })

        # Simulación paso a paso de la ruta crítica P05 -> V_PG1 (30.0 kg)
        sim_p05 = SimuladorCadenaFrio(grafo, tiempo_limite_h=TIEMPO_MAX_SIN_FRIO_H)
        ruta_p05 = ["P05", "A03", "D01", "M02", "A01", "F02", "V_PG1"]
        res_p05 = sim_p05.simular_ruta(ruta_p05, 30.0)
        pasos_p05 = []
        det_prev = 0.0
        kg_prev = 30.0
        tsf_prev = 0.0
        for idx_s, est in enumerate(res_p05.estados):
            nd = grafo.obtener_nodo(est.nodo_id)
            ar_prev = grafo.obtener_arista(ruta_p05[idx_s - 1], est.nodo_id) if idx_s > 0 else None
            t_arista = round(ar_prev.tiempo_h, 2) if ar_prev else 0.0
            t_tramo_ef = round(t_arista + (TIEMPO_TRANSBORDO_MUELLE_H if (nd and nd.tipo == "M") else 0.0), 2)
            tsf_antes_reset = round(tsf_prev + t_tramo_ef, 2) if idx_s > 0 else 0.0
            delta_det = round(est.deterioro_pct - det_prev, 3)
            kg_perd_paso = round(kg_prev - est.kg_actual, 3)
            cap_nd = int(nd.capacidad_frio_kg) if nd else 0
            if idx_s == 0:
                regla_mat = "Inicialización: kg_actual = 30.000 kg, temp = 30.0 °C, tiempo_sin_frio = 0.00 h."
            elif nd and nd.tipo == "D":
                regla_mat = f"calcular_deterioro({t_arista:.2f}, 30°C) = {t_arista*2.5:.3f}% × FACTOR_DETERIORO_ZONA_D ({FACTOR_DETERIORO_ZONA_D:g}) = {delta_det:.3f}%."
            elif nd and nd.tipo == "M":
                regla_mat = f"t_tramo = {t_arista:.2f} + TIEMPO_TRANSBORDO_MUELLE_H (0.25) = {t_tramo_ef:.2f} h → calcular_deterioro({t_tramo_ef:.2f}, 30°C) = {delta_det:.3f}%."
            elif est.refrigerado:
                regla_mat = f"Tramo a 30°C ({t_arista:.2f} h × 2.5%/h = {delta_det:.3f}%). Luego tiene_frio=(tipo='{nd.tipo}'∈{{F,A}} ∧ cap={cap_nd}>0)=True → temp=0°C y reset t_sin_frío=0.00 h."
            else:
                regla_mat = f"Tramo final a 30°C: {t_arista:.2f} h × 2.5%/h = {delta_det:.3f}%. Evaluación final: t_sin_frío ({est.tiempo_sin_frio_h:.2f} h) ≤ 4.00 h → cumple_limite=True."
            pasos_p05.append({
                "paso": idx_s,
                "arista_id": f"{ruta_p05[idx_s - 1]} → {est.nodo_id}" if idx_s > 0 else "Inicio",
                "nodo_id": est.nodo_id,
                "nombre": nd.nombre if nd else est.nodo_id,
                "tipo": nd.tipo if nd else "",
                "capacidad_frio_kg": cap_nd,
                "tiempo_arista_h": t_arista,
                "tiempo_tramo_ef_h": t_tramo_ef,
                "tiempo_acum_h": round(est.tiempo_acumulado_h, 2),
                "tsf_antes_reset_h": tsf_antes_reset,
                "tiempo_sin_frio_h": round(est.tiempo_sin_frio_h, 2),
                "temp_c": round(est.temperatura_actual_c, 1),
                "delta_det_pct": delta_det,
                "det_acum_pct": round(est.deterioro_pct, 3),
                "kg_anterior": round(kg_prev, 3),
                "kg_actual": round(est.kg_actual, 3),
                "kg_perdidos_paso": kg_perd_paso,
                "refrigerado": bool(est.refrigerado),
                "efecto": est.efecto or ("Inicio" if idx_s == 0 else "Entrega"),
                "regla_matematica": regla_mat,
            })
            det_prev = est.deterioro_pct
            kg_prev = est.kg_actual
            tsf_prev = est.tiempo_sin_frio_h

        # Sustitución real arista F02 -> F03
        ar_f23 = grafo.obtener_arista("F02", "F03")
        f23_info = {}
        if ar_f23:
            f23_info = {
                "id": "F02 → F03",
                "distancia_km": round(ar_f23.distancia_km, 1),
                "tiempo_h": round(ar_f23.tiempo_h, 2),
                "factor_estado_via": round(ar_f23.factor_estado_via, 2),
                "factor_flujo": round(ar_f23.factor_flujo, 2),
                "tiempo_espera_h": round(ar_f23.tiempo_espera_h, 2),
                "tiempo_efectivo_h": round(ar_f23.tiempo_efectivo_h, 3),
                "t_norm": round(ar_f23.tiempo_efectivo_h / 48.0, 5),
                "costo_cop": int(ar_f23.costo_cop),
                "costo_peajes_cop": int(ar_f23.costo_peajes_cop),
                "costo_total_cop": int(ar_f23.costo_total_cop),
                "c_norm": round(ar_f23.costo_total_cop / 5_000_000.0, 5),
                "riesgo_termico": round(ar_f23.riesgo_termico, 2),
                "riesgo_accidente": round(ar_f23.riesgo_accidente, 2),
                "riesgo_derrumbe": round(ar_f23.riesgo_derrumbe, 2),
                "riesgo_combinado": round(ar_f23.riesgo_combinado, 4),
                "perdida_esperada": round(ar_f23.perdida_esperada, 3),
                "p_norm": round(ar_f23.perdida_esperada / 100.0, 5),
                "confiabilidad_via": round(ar_f23.confiabilidad_via, 2),
                "prob_retraso": round(ar_f23.prob_retraso, 2),
                "peso_combinado": round(ar_f23.to_dict()["peso_combinado"], 4),
            }

        flp_detallado = {
            "oferta_total_kg": oferta_tot,
            "num_centros_csv": len([n for n in grafo.nodos if n.tipo in ("F", "A", "M", "D", "H")]),
            "candidatos": candidatos_info,
            "matriz_7x5": matriz_7x5,
            "simulacion_p05": {
                "ruta": ruta_p05,
                "kg_inicial": round(res_p05.kg_inicial, 3),
                "kg_final": round(res_p05.kg_final, 3),
                "perdida_kg": round(res_p05.perdida_kg, 3),
                "perdida_pct": round(res_p05.perdida_pct, 2),
                "costo_perdida_cop": round(res_p05.costo_perdida_cop, 0),
                "tiempo_total_h": round(res_p05.tiempo_total_h, 2),
                "tiempo_sin_frio_final_h": round(res_p05.tiempo_sin_frio_total_h, 2),
                "cumple_limite": bool(res_p05.cumple_limite),
                "pasos": pasos_p05,
            },
            "arista_f02_f03": f23_info,
        }
    except Exception:
        flp_detallado = {}

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
        "flp_detallado": flp_detallado,
        "escenarios_pg": escenarios_pg,
        "escenarios_bog": escenarios_bog,
        "eca": eca_data,
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
            "COSTO_FIJO_CENTRO_FRIO": COSTO_FIJO_CENTRO_FRIO,
            "TEMP_AMBIENTE_LLANOS_C": TEMP_AMBIENTE_LLANOS_C,
            "VIDA_UTIL_HORAS": VIDA_UTIL_HORAS,
            "COSTO_PERDIDA_KG": COSTO_PERDIDA_KG,
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
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Newsreader:ital,opsz,wght@0,6..72,400;1,6..72,400;1,6..72,500&display=swap');

  :root {{
    /* ═══════════════════════════════════════════════════════════════
       TOKENS CROMÁTICOS: DALA × APPLE MIDNIGHT × ORYZO EDITORIAL
       ═══════════════════════════════════════════════════════════════ */
    --color-void: #000000;
    --color-obsidian: #000000;
    --color-graphite: #0a0a0c;
    --color-carbon: #121316;
    --color-walnut-shadow: #100904;
    --color-bark-brown: #382416;
    --color-steel: #24252a;
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
    --color-electric-iris: #8052ff;
    --color-saffron-spark: #ffb829;
    --color-deep-verdant: #15846e;
    --color-apple-blue: #0071e3;
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

    /* Familias tipográficas refinadas con Google Fonts garantizadas */
    --font-display: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --font-serif: 'Newsreader', 'Georgia', 'Times New Roman', serif;
    --font-mono: 'JetBrains Mono', ui-monospace, 'SF Mono', 'Cascadia Code', Consolas, monospace;

    /* Radios y geometría (Dala 24px pills / Apple 28px plinths) */
    --radius-cards: 24px;
    --radius-panel: 20px;
    --radius-nav: 18px;
    --radius-pill: 9999px;

    /* Curvas Emil Kowalski (animate & review-animations) */
    --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
    --ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
    --ease-drawer: cubic-bezier(0.32, 0.72, 0, 1);
    --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
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
  body:not(.folium-embed-only) #folium-frio-card {{
    top: 146px !important;
  }}
  body:not(.folium-embed-only) #folium-itin-card {{
    top: 295px !important;
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
    font-size: 15.5px;
    line-height: 1.6;
    color: var(--color-porcelain);
    background-color: var(--color-void);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
  }}
  h1 {{
    font-family: var(--font-display);
    font-size: 48px;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.035em;
    color: var(--color-bone-white);
    margin: 0;
  }}
  h2 {{
    font-family: var(--font-display);
    font-size: 32px;
    font-weight: 600;
    line-height: 1.12;
    letter-spacing: -0.025em;
    color: var(--color-porcelain);
    margin: 0;
  }}
  h3 {{
    font-family: var(--font-display);
    font-size: 21px;
    font-weight: 600;
    line-height: 1.25;
    letter-spacing: -0.015em;
    color: var(--color-porcelain);
    margin: 0;
  }}
  p {{
    font-size: 15.5px;
    font-weight: 400;
    line-height: 1.65;
    color: var(--color-silver-mist);
    margin: 0;
    text-wrap: pretty;
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
    font-family: var(--font-display);
    font-size: 50px;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.038em;
    color: var(--color-bone-white);
    margin: 0 0 18px 0;
  }}
  .agy-hero-title em {{
    font-family: var(--font-serif);
    font-style: italic;
    font-weight: 400;
    color: var(--color-saffron-spark);
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
    background: linear-gradient(135deg, var(--color-electric-iris) 0%, #a855f7 100%);
    color: var(--color-bone-white);
    border: none;
    border-radius: var(--radius-pill);
    padding: 14px 26px;
    font-family: var(--font-sans);
    font-size: 13.5px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 4px 20px rgba(128, 82, 255, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.25);
    transition: transform 150ms var(--ease-out), box-shadow 150ms var(--ease-out), filter 150ms var(--ease-out);
  }}
  .agy-pill-cta:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(128, 82, 255, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.4);
    filter: brightness(1.08);
  }}
  .agy-pill-cta:active {{
    transform: scale(0.96) translateY(0);
    box-shadow: 0 2px 10px rgba(128, 82, 255, 0.3);
  }}
  .agy-pill-ghost {{
    background: rgba(255, 255, 255, 0.03);
    color: var(--color-porcelain);
    border: 1px solid var(--color-steel);
    border-radius: var(--radius-pill);
    padding: 13px 22px;
    font-family: var(--font-sans);
    font-size: 13.5px;
    font-weight: 600;
    letter-spacing: 0.02em;
    cursor: pointer;
    transition: transform 150ms var(--ease-out), border-color 150ms var(--ease-out), background-color 150ms var(--ease-out);
  }}
  .agy-pill-ghost:hover {{
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(255, 255, 255, 0.25);
    transform: translateY(-2px);
  }}
  .agy-pill-ghost:active {{
    transform: scale(0.96) translateY(0);
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
    background: radial-gradient(400px circle at var(--mouse-x, 50%) var(--mouse-y, 50%), rgba(128, 82, 255, 0.08), transparent 70%), linear-gradient(180deg, #17181c 0%, #0a0a0c 100%);
    border: 1px solid var(--color-steel);
    border-radius: var(--radius-cards);
    padding: 24px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 12px;
    position: relative;
    overflow: hidden;
    transition: transform 220ms var(--ease-out), border-color 220ms var(--ease-out), box-shadow 220ms var(--ease-out);
  }}
  .agy-plinth-card::before {{
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    padding: 1px;
    background: radial-gradient(350px circle at var(--mouse-x, 50%) var(--mouse-y, 50%), rgba(255, 184, 41, 0.45), transparent 70%);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
    opacity: 0;
    transition: opacity 250ms var(--ease-out);
  }}
  .agy-plinth-card:hover {{
    transform: translateY(-3px);
    border-color: rgba(255, 184, 41, 0.35);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45), 0 0 24px rgba(128, 82, 255, 0.12);
  }}
  .agy-plinth-card:hover::before {{
    opacity: 1;
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
    font-family: var(--font-display);
    font-size: 42px;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.04em;
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
    font-family: 'Newsreader', 'Cambria Math', 'Latin Modern Math', 'Times New Roman', Georgia, serif;
    font-size: 21px;
    line-height: 1.35;
    color: var(--color-bone-white);
    display: inline-block;
    letter-spacing: 0.015em;
    white-space: nowrap;
  }}
  .agy-math-eq var {{
    font-family: 'Newsreader', 'Cambria Math', 'Times New Roman', Georgia, serif;
    font-style: italic;
    font-weight: 600;
    color: var(--color-bone-white);
  }}
  .agy-math-eq sub {{
    font-family: 'Newsreader', 'Cambria Math', 'Times New Roman', Georgia, serif;
    font-style: normal;
    font-size: 0.72em;
    font-weight: 600;
    color: var(--color-silver-mist);
    vertical-align: -0.22em;
    margin-left: 1px;
    margin-right: 3px;
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
    gap: 8px;
  }}
  .agy-prop-group h4.agy-prop-h4-btn {{
    cursor: pointer;
    padding: 6px 8px;
    margin: -4px -8px 10px -8px;
    border-radius: 10px;
    border: 1px solid transparent;
    border-bottom: 1px dashed var(--color-cork-border);
    transition: background 150ms ease, border-color 150ms ease, color 150ms ease;
  }}
  .agy-prop-group h4.agy-prop-h4-btn:hover,
  .agy-prop-group h4.agy-prop-h4-btn:focus-visible {{
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.12);
    outline: none;
  }}
  .agy-prop-h4-pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 10.5px;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.11);
    color: var(--color-porcelain);
    letter-spacing: 0.03em;
    white-space: nowrap;
    flex-shrink: 0;
  }}
  .agy-prop-group h4.agy-prop-h4-btn:hover .agy-prop-h4-pill {{
    background: rgba(255, 255, 255, 0.12);
    border-color: rgba(255, 255, 255, 0.24);
  }}
  /* Modal de Variables y Fórmulas (limpio, aireado y monocromático-editorial) */
  .agy-vm-switch {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 0 0 12px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }}
  .agy-vm-entity-btn {{
    padding: 6px 12px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.10);
    background: rgba(255, 255, 255, 0.03);
    color: var(--color-ash-gray);
    font-family: var(--font-mono);
    font-size: 12px;
    cursor: pointer;
    transition: all 140ms ease;
  }}
  .agy-vm-entity-btn.active {{
    background: rgba(255, 255, 255, 0.11);
    border-color: rgba(255, 255, 255, 0.28);
    color: var(--color-porcelain);
    font-weight: 600;
  }}
  .agy-vm-tabs {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 16px;
  }}
  .agy-vm-tab {{
    padding: 5px 10px;
    border-radius: 7px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: transparent;
    color: var(--color-ash-gray);
    font-size: 12px;
    cursor: pointer;
    transition: all 140ms ease;
  }}
  .agy-vm-tab:hover {{
    color: var(--color-porcelain);
    border-color: rgba(255, 255, 255, 0.18);
  }}
  .agy-vm-tab.active {{
    background: #242428;
    border-color: rgba(255, 255, 255, 0.25);
    color: var(--color-porcelain);
    font-weight: 600;
  }}
  .agy-vm-lead {{
    font-size: 13.5px;
    line-height: 1.65;
    color: var(--color-silver-mist);
    margin: 0 0 16px 0;
    padding: 12px 14px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
  }}
  .agy-vm-list {{
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .agy-vm-card {{
    background: #161618;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .agy-vm-card-top {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }}
  .agy-vm-name {{
    font-family: var(--font-mono);
    font-size: 13.5px;
    font-weight: 700;
    color: var(--color-porcelain);
  }}
  .agy-vm-badges {{
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
  }}
  .agy-vm-tag {{
    font-family: var(--font-mono);
    font-size: 10.5px;
    padding: 2px 7px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.09);
    color: var(--color-silver-mist);
  }}
  .agy-vm-tag.calc {{
    background: rgba(255, 255, 255, 0.09);
    border-color: rgba(255, 255, 255, 0.22);
    color: var(--color-porcelain);
  }}
  .agy-vm-desc {{
    font-size: 13.5px;
    line-height: 1.6;
    color: var(--color-silver-mist);
    margin: 0;
  }}
  .agy-vm-meta {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 8px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--color-ash-gray);
    padding-top: 6px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
  }}
  .agy-vm-live {{
    color: var(--color-porcelain);
    font-variant-numeric: tabular-nums;
  }}
  .agy-vm-formula {{
    margin-top: 2px;
    padding: 12px 16px;
    border-radius: 10px;
    background: #0d0d10;
    border: 1px solid rgba(255, 255, 255, 0.09);
    overflow-x: auto;
  }}
  .agy-vm-formula-cap {{
    font-family: var(--font-mono);
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-ash-gray);
    margin-bottom: 10px;
  }}
  .agy-vm-math-stack {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
  }}
  .agy-vm-math-row {{
    display: inline-flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 4px;
    font-family: 'Newsreader', 'Cambria Math', 'Latin Modern Math', 'Times New Roman', Georgia, serif;
    font-size: 17px;
    line-height: 1.45;
    color: var(--color-bone-white);
  }}
  .agy-vm-math-row var {{
    font-family: 'Newsreader', 'Cambria Math', 'Times New Roman', Georgia, serif;
    font-style: italic;
    font-weight: 600;
    color: var(--color-bone-white);
  }}
  .agy-vm-math-row sub {{
    font-family: 'Newsreader', 'Cambria Math', 'Times New Roman', Georgia, serif;
    font-style: normal;
    font-weight: 600;
    font-size: 0.72em;
    color: var(--color-silver-mist);
    vertical-align: -0.22em;
    margin-left: 1px;
  }}
  .agy-vm-math-row sub i {{
    font-style: italic;
  }}
  .agy-vm-math-row sup {{
    font-family: 'Newsreader', 'Cambria Math', 'Times New Roman', Georgia, serif;
    font-style: normal;
    font-weight: 600;
    font-size: 0.72em;
    vertical-align: 0.45em;
    margin-left: 1px;
  }}
  .agy-vm-math-row code {{
    font-family: var(--font-mono);
    font-size: 0.82em;
    background: rgba(255, 255, 255, 0.07);
    padding: 1px 6px;
    border-radius: 5px;
    color: var(--color-porcelain);
  }}
  .agy-vm-math-sub {{
    font-size: 13.5px;
    color: var(--color-silver-mist);
  }}
  .agy-vm-math-sep {{
    display: inline-block;
    width: 1px;
    height: 16px;
    background: rgba(255, 255, 255, 0.16);
    margin: 0 10px;
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

  /* ═══════════════════════════════════════════════════════════════
     COMPONENTES AVANZADOS: MODAL CARDS (MORPH), SIMULADOR CROSSHAIR,
     PESTAÑAS DE CONSTRUCCIÓN DE GRAFOS Y COMPARADOR ANTES VS DESPUÉS
     ═══════════════════════════════════════════════════════════════ */
  .agy-modal-trigger {{
    margin-top: 8px;
    align-self: flex-start;
    background: rgba(128, 82, 255, 0.12);
    color: var(--color-bone-white);
    border: 1px solid rgba(128, 82, 255, 0.4);
    border-radius: 9999px;
    padding: 6px 13px;
    font-family: var(--font-sans);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: transform 160ms var(--ease-out), background-color 160ms var(--ease-out), border-color 160ms var(--ease-out);
  }}
  .agy-modal-trigger:hover {{
    background: var(--color-electric-iris);
    border-color: var(--color-bone-white);
    transform: translateY(-1px);
  }}
  .agy-modal-trigger:active {{
    transform: scale(0.96);
  }}

  #agy-modal-backdrop {{
    position: fixed;
    inset: 0;
    z-index: 20000;
    background: rgba(0, 0, 0, 0.82);
    backdrop-filter: blur(14px);
    display: none;
    place-items: center;
    padding: 24px;
    opacity: 0;
    transition: opacity 220ms var(--ease-out);
  }}
  #agy-modal-backdrop.open {{
    display: grid;
    opacity: 1;
  }}
  .agy-modal-dialog {{
    width: min(920px, 96vw);
    max-height: 86vh;
    overflow-y: auto;
    background: linear-gradient(180deg, #14151a 0%, #08080a 100%);
    border: 1px solid rgba(128, 82, 255, 0.55);
    border-radius: 24px;
    padding: 30px 32px;
    color: var(--color-porcelain);
    transform: scale(0.92) translateY(14px);
    transition: transform 260ms var(--ease-out);
    position: relative;
  }}
  #agy-modal-backdrop.open .agy-modal-dialog {{
    transform: scale(1) translateY(0);
  }}
  .agy-modal-close {{
    position: sticky;
    top: 0;
    float: right;
    width: 36px;
    height: 36px;
    border-radius: 9999px;
    border: 1px solid var(--color-steel);
    background: rgba(20, 21, 26, 0.9);
    color: var(--color-bone-white);
    font-size: 18px;
    cursor: pointer;
    display: grid;
    place-items: center;
    transition: transform 150ms var(--ease-out), background-color 150ms var(--ease-out);
  }}
  .agy-modal-close:hover {{
    background: var(--color-electric-iris);
    transform: rotate(90deg);
  }}

  /* ═══════════════════════════════════════════════════════════════
     SIMULADOR TÉRMICO v2 — Estilo Highcharts (Spline + Plot Bands)
     ═══════════════════════════════════════════════════════════════ */
  .agy-crosshair-panel {{
    background: linear-gradient(180deg, #0d0e12 0%, #060608 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 20px;
    padding: 28px 32px 24px;
    margin-bottom: 28px;
    position: relative;
  }}

  /* Encabezado del simulador */
  .agy-chart-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 20px;
  }}
  .agy-chart-header h2 {{
    font-family: var(--font-sans);
    font-size: 18px;
    font-weight: 700;
    color: var(--color-porcelain);
    margin: 0;
    line-height: 1.35;
  }}
  .agy-chart-header p {{
    font-family: var(--font-sans);
    font-size: 13px;
    color: var(--color-ash);
    margin: 4px 0 0;
    line-height: 1.5;
    max-width: 620px;
  }}

  /* Leyenda interactiva tipo Highcharts */
  .agy-chart-legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    justify-content: center;
    margin: 0 0 14px;
  }}
  .agy-legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 8px;
    font-family: var(--font-sans);
    font-size: 12px;
    font-weight: 500;
    color: var(--color-silver-mist);
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    cursor: pointer;
    transition: opacity 160ms var(--ease-out), background 160ms var(--ease-out);
    user-select: none;
  }}
  .agy-legend-item:hover {{
    background: rgba(255, 255, 255, 0.07);
  }}
  .agy-legend-item.disabled {{
    opacity: 0.35;
  }}
  .agy-legend-swatch {{
    width: 14px;
    height: 3px;
    border-radius: 2px;
    flex-shrink: 0;
  }}

  /* Contenedor del canvas del gráfico */
  .agy-chart-canvas-wrap {{
    position: relative;
    width: 100%;
    border-radius: 14px;
    overflow: hidden;
    background: #08080b;
    border: 1px solid rgba(255, 255, 255, 0.06);
  }}
  .agy-chart-canvas-wrap canvas {{
    display: block;
    width: 100%;
    cursor: crosshair;
  }}

  /* Tooltip flotante tipo Highcharts */
  .agy-chart-tooltip {{
    position: absolute;
    pointer-events: none;
    z-index: 100;
    background: rgba(10, 10, 14, 0.94);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    padding: 10px 14px;
    font-family: var(--font-sans);
    font-size: 12px;
    color: var(--color-porcelain);
    min-width: 200px;
    opacity: 0;
    transform: translateY(4px);
    transition: opacity 120ms var(--ease-out), transform 120ms var(--ease-out);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
  }}
  .agy-chart-tooltip.visible {{
    opacity: 1;
    transform: translateY(0);
  }}
  .agy-chart-tooltip .tt-title {{
    font-weight: 700;
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--color-bone-white);
    margin-bottom: 6px;
    padding-bottom: 5px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }}
  .agy-chart-tooltip .tt-row {{
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 2px 0;
    font-variant-numeric: tabular-nums;
  }}
  .agy-chart-tooltip .tt-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    border: 1.5px solid rgba(255, 255, 255, 0.5);
  }}
  .agy-chart-tooltip .tt-val {{
    margin-left: auto;
    font-weight: 600;
    font-family: var(--font-mono);
  }}

  /* Controles del slider y presets */
  .agy-crosshair-controls {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 14px;
    margin: 16px 0 0;
    padding: 14px 18px;
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
  }}
  .agy-time-slider {{
    flex: 1;
    min-width: 200px;
    -webkit-appearance: none;
    appearance: none;
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    outline: none;
    cursor: grab;
  }}
  .agy-time-slider::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--color-electric-iris);
    border: 2px solid var(--color-bone-white);
    cursor: grab;
    box-shadow: 0 0 10px rgba(128, 82, 255, 0.5);
    transition: transform 120ms var(--ease-out), box-shadow 120ms var(--ease-out);
  }}
  .agy-time-slider::-webkit-slider-thumb:hover {{
    transform: scale(1.15);
    box-shadow: 0 0 16px rgba(128, 82, 255, 0.7);
  }}
  .agy-time-slider:active {{
    cursor: grabbing;
  }}
  .agy-preset-btns {{
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }}
  .agy-preset-btn {{
    background: rgba(255, 255, 255, 0.04);
    color: var(--color-silver-mist);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 6px 12px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 140ms var(--ease-out);
    white-space: nowrap;
  }}
  .agy-preset-btn:hover {{
    background: rgba(128, 82, 255, 0.15);
    border-color: rgba(128, 82, 255, 0.4);
    color: var(--color-bone-white);
  }}
  .agy-preset-btn:active {{
    transform: scale(0.96);
  }}
  .agy-preset-btn.active-preset {{
    background: var(--color-electric-iris);
    color: var(--color-bone-white);
    border-color: rgba(255, 255, 255, 0.2);
  }}

  /* Tarjetas de readout mejoradas */
  .agy-sim-readout-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
    margin-top: 16px;
  }}
  .agy-readout-card {{
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 16px 18px;
    transition: border-color 200ms var(--ease-out);
  }}
  .agy-readout-card:hover {{
    border-color: rgba(255, 255, 255, 0.14);
  }}
  .agy-readout-val {{
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 26px;
    font-weight: 700;
    line-height: 1.1;
  }}
  .agy-readout-label {{
    font-family: var(--font-sans);
    font-size: 12px;
    font-weight: 600;
    color: var(--color-ash);
    margin-top: 6px;
    letter-spacing: 0.03em;
  }}
  .agy-readout-sub {{
    font-family: var(--font-sans);
    font-size: 11px;
    color: var(--color-slate);
    margin-top: 4px;
    line-height: 1.4;
  }}

  /* Fórmula interactiva grande y panel de desglose matemático */
  .agy-therm-formula-btn {{
    display: inline-flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 5px;
    padding: 12px 20px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(128, 82, 255, 0.14) 0%, rgba(255, 184, 41, 0.08) 100%);
    border: 1px solid rgba(128, 82, 255, 0.45);
    color: var(--color-bone-white);
    cursor: pointer;
    user-select: none;
    transition: transform 180ms cubic-bezier(0.16, 1, 0.3, 1), border-color 180ms var(--ease-out), background 180ms var(--ease-out), box-shadow 180ms var(--ease-out);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
  }}
  .agy-therm-formula-btn:hover {{
    transform: translateY(-2px);
    border-color: var(--color-saffron-spark);
    background: linear-gradient(135deg, rgba(128, 82, 255, 0.24) 0%, rgba(255, 184, 41, 0.14) 100%);
    box-shadow: 0 8px 28px rgba(128, 82, 255, 0.25);
  }}
  .agy-therm-formula-btn:active {{
    transform: scale(0.97);
  }}
  .agy-therm-formula-btn.is-open {{
    border-color: var(--color-saffron-spark);
    background: linear-gradient(135deg, rgba(128, 82, 255, 0.28) 0%, rgba(255, 184, 41, 0.18) 100%);
  }}
  .agy-therm-formula-main {{
    font-family: var(--font-serif);
    font-size: 24px;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--color-bone-white);
    display: flex;
    align-items: center;
    gap: 8px;
    line-height: 1.2;
  }}
  .agy-therm-formula-main var {{
    font-style: italic;
    color: var(--color-saffron-spark);
    padding: 2px 5px;
    border-radius: 6px;
    transition: background 150ms var(--ease-out), color 150ms var(--ease-out);
  }}
  .agy-therm-formula-btn:hover .agy-therm-formula-main var {{
    background: rgba(255, 184, 41, 0.14);
  }}
  .agy-therm-formula-hint {{
    font-family: var(--font-mono);
    font-size: 10.5px;
    font-weight: 600;
    color: var(--color-silver-mist);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}
  .agy-therm-formula-hint .hint-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--color-saffron-spark);
    box-shadow: 0 0 8px var(--color-saffron-spark);
  }}

  /* Panel desplegable interactivo de la ecuación */
  .agy-formula-drawer {{
    display: none;
    margin: 0 0 18px;
    padding: 20px 22px;
    border-radius: 16px;
    background: rgba(14, 15, 22, 0.92);
    border: 1px solid rgba(128, 82, 255, 0.35);
    animation: agySlideEnter 240ms cubic-bezier(0.16, 1, 0.3, 1) both;
  }}
  .agy-formula-drawer.open {{
    display: block;
  }}
  .agy-formula-terms-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
    padding-bottom: 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
  }}
  .agy-term-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 14px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: var(--color-porcelain);
    font-family: var(--font-sans);
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 150ms var(--ease-out);
  }}
  .agy-term-pill:hover {{
    background: rgba(128, 82, 255, 0.16);
    border-color: rgba(128, 82, 255, 0.45);
  }}
  .agy-term-pill.active {{
    background: var(--color-electric-iris);
    border-color: rgba(255, 255, 255, 0.3);
    color: var(--color-bone-white);
  }}
  .agy-term-pill code {{
    font-family: var(--font-serif);
    font-style: italic;
    font-size: 15px;
    font-weight: 700;
    color: var(--color-saffron-spark);
    background: rgba(0, 0, 0, 0.35);
    padding: 2px 7px;
    border-radius: 6px;
  }}
  .agy-term-pill.active code {{
    color: #ffffff;
  }}
  .agy-formula-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
  }}
  .agy-formula-card {{
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 16px 18px;
    transition: border-color 180ms var(--ease-out), background 180ms var(--ease-out), transform 180ms var(--ease-out);
    cursor: pointer;
  }}
  .agy-formula-card:hover {{
    border-color: rgba(255, 184, 41, 0.4);
    background: rgba(255, 255, 255, 0.04);
  }}
  .agy-formula-card.highlighted {{
    border-color: var(--color-saffron-spark);
    background: linear-gradient(180deg, rgba(128, 82, 255, 0.12) 0%, rgba(255, 184, 41, 0.05) 100%);
    transform: translateY(-2px);
  }}
  .agy-fcard-head {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 8px;
  }}
  .agy-fcard-sym {{
    font-family: var(--font-serif);
    font-style: italic;
    font-size: 22px;
    font-weight: 700;
    color: var(--color-saffron-spark);
    background: rgba(255, 184, 41, 0.1);
    border: 1px solid rgba(255, 184, 41, 0.28);
    padding: 3px 11px;
    border-radius: 8px;
  }}
  .agy-fcard-title {{
    font-family: var(--font-sans);
    font-size: 13.5px;
    font-weight: 700;
    color: var(--color-bone-white);
    flex: 1;
  }}
  .agy-fcard-body {{
    font-family: var(--font-sans);
    font-size: 12.5px;
    color: var(--color-silver-mist);
    line-height: 1.55;
  }}
  .agy-fcard-body ul {{
    margin: 6px 0 0;
    padding-left: 16px;
  }}
  .agy-fcard-body li {{
    margin-bottom: 4px;
  }}
  .agy-fcard-mathbox {{
    margin-top: 10px;
    padding: 12px 16px;
    border-radius: 10px;
    background: rgba(8, 8, 12, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.08);
    font-family: var(--font-serif);
    font-size: 16.5px;
    color: var(--color-porcelain);
    line-height: 1.75;
    letter-spacing: 0.015em;
  }}

  /* ═══════════════════════════════════════════════════════════════
     TIPOGRAFÍA MATEMÁTICA EDITORIAL (KaTeX / LaTeX Display Style)
     Para todas las fórmulas, fracciones, sumatorias y subíndices
     ═══════════════════════════════════════════════════════════════ */
  .agy-tex {{
    font-family: 'Newsreader', 'Cambria Math', 'Latin Modern Math', 'Times New Roman', Georgia, serif;
    font-size: 1.08em;
    font-weight: 500;
    color: var(--color-bone-white);
    letter-spacing: 0.015em;
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 5px;
    vertical-align: middle;
  }}
  .agy-tex-block {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin: 10px 0;
    padding: 14px 20px;
    border-radius: 12px;
    background: rgba(10, 10, 15, 0.78);
    border: 1px solid rgba(255, 255, 255, 0.09);
    font-family: 'Newsreader', 'Cambria Math', 'Latin Modern Math', 'Times New Roman', Georgia, serif;
    font-size: 18px;
    color: var(--color-bone-white);
    overflow-x: auto;
  }}
  .agy-tex var, .agy-tex-block var, .agy-fcard-mathbox var {{
    font-family: 'Newsreader', 'Cambria Math', 'Times New Roman', Georgia, serif;
    font-style: italic;
    font-weight: 600;
    color: var(--color-bone-white);
  }}
  .agy-tex sub, .agy-tex-block sub, .agy-fcard-mathbox sub,
  .agy-therm-formula-main sub, .agy-fcard-sym sub, .agy-term-pill code sub {{
    font-family: 'Newsreader', 'Cambria Math', 'Times New Roman', Georgia, serif;
    font-style: normal;
    font-weight: 600;
    font-size: 0.72em;
    color: var(--color-silver-mist);
    vertical-align: -0.22em;
    margin-left: 1px;
  }}
  .agy-tex sub i, .agy-tex-block sub i, .agy-fcard-mathbox sub i,
  .agy-therm-formula-main sub i, .agy-fcard-sym sub i {{
    font-style: italic;
  }}
  .agy-tex sup, .agy-tex-block sup, .agy-fcard-mathbox sup {{
    font-family: 'Newsreader', 'Cambria Math', 'Times New Roman', Georgia, serif;
    font-style: normal;
    font-weight: 600;
    font-size: 0.72em;
    vertical-align: 0.45em;
    margin-left: 1px;
  }}
  .agy-sum {{
    font-family: 'Cambria Math', 'Times New Roman', serif;
    font-style: normal;
    font-size: 1.45em;
    font-weight: 400;
    line-height: 1;
    vertical-align: -0.08em;
    margin: 0 2px;
    color: var(--color-bone-white);
  }}
  .agy-frac {{
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    vertical-align: middle;
    margin: 0 5px;
    font-size: 0.92em;
    line-height: 1.15;
  }}
  .agy-frac-num {{
    border-bottom: 1.3px solid rgba(255, 255, 255, 0.75);
    padding: 0 6px 3px;
    text-align: center;
    white-space: nowrap;
  }}
  .agy-frac-den {{
    padding: 3px 6px 0;
    text-align: center;
    white-space: nowrap;
  }}
  .agy-op {{
    font-style: normal;
    color: var(--color-silver-mist);
    margin: 0 3px;
  }}
  .agy-step-tabs {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 18px;
  }}
  .agy-step-tab {{
    background: rgba(255, 255, 255, 0.03);
    color: var(--color-silver-mist);
    border: 1px solid var(--color-steel);
    border-radius: 9999px;
    padding: 9px 16px;
    font-family: var(--font-sans);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 160ms var(--ease-out);
  }}
  .agy-step-tab.active {{
    background: var(--color-electric-iris);
    color: var(--color-bone-white);
    border-color: rgba(255, 255, 255, 0.25);
  }}
  .agy-step-pane {{
    display: none;
    animation: agySlideEnter 220ms var(--ease-out) both;
  }}
  .agy-step-pane.active {{
    display: block;
  }}

  /* Comparador Interactivo Antes vs. Después */
  .agy-comp-card {{
    background: linear-gradient(180deg, #15161b 0%, #09090c 100%);
    border: 1px solid var(--color-steel);
    border-radius: 20px;
    padding: 20px 22px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .agy-comp-dual-bar {{
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .agy-comp-track {{
    height: 10px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 9999px;
    overflow: hidden;
  }}
  .agy-comp-fill-before {{
    height: 100%;
    background: linear-gradient(90deg, #dc5000, #ff6b2b);
    border-radius: 9999px;
    transition: width 550ms var(--ease-out);
  }}
  .agy-comp-fill-after {{
    height: 100%;
    background: linear-gradient(90deg, #15846e, #10b981);
    border-radius: 9999px;
    transition: width 550ms var(--ease-out);
  }}
  .agy-delta-pill {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 700;
    background: rgba(16, 185, 129, 0.14);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.35);
  }}

  /* ═══════════════════════════════════════════════════════════════
     PASO 2 & PASO 3 · WORKSPACE DEDICADO FLP + TOOLTIP TÉCNICO HOVER
     ═══════════════════════════════════════════════════════════════ */
  .agy-flp-subtabs {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0 0 20px 0;
    padding: 6px;
    background: #121216;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
  }}
  .agy-flp-subtab {{
    background: transparent;
    color: #a1a1aa;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 9px 14px;
    font-family: var(--font-sans);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: background 150ms ease, color 150ms ease, border-color 150ms ease;
  }}
  .agy-flp-subtab:hover {{
    color: #f5f5f7;
    background: rgba(255, 255, 255, 0.05);
  }}
  .agy-flp-subtab.active {{
    background: rgba(128, 82, 255, 0.22);
    color: #f5f5f7;
    border-color: rgba(128, 82, 255, 0.55);
  }}
  .agy-flp-pane {{
    display: none;
  }}
  .agy-flp-pane.active {{
    display: block;
    animation: agySlideEnter 180ms var(--ease-out) both;
  }}
  .agy-flp-pipe {{
    display: grid;
    grid-template-columns: 1fr auto 1fr auto 1.2fr;
    gap: 12px;
    align-items: stretch;
    margin-bottom: 18px;
  }}
  @media (max-width: 980px) {{
    .agy-flp-pipe {{
      grid-template-columns: 1fr;
    }}
    .agy-flp-pipe-arrow {{
      display: none;
    }}
  }}
  .agy-flp-pipe-box {{
    background: #141418;
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 12px;
    padding: 14px 16px;
  }}
  .agy-flp-pipe-box.accent {{
    border-color: rgba(128, 82, 255, 0.4);
    background: #171521;
  }}
  .agy-flp-pipe-arrow {{
    display: flex;
    align-items: center;
    justify-content: center;
    color: #8052ff;
    font-size: 20px;
    font-weight: 700;
  }}
  .agy-flp-pipe-kicker {{
    font-family: var(--font-mono);
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #a1a1aa;
    margin-bottom: 6px;
  }}
  .agy-flp-pipe-title {{
    font-size: 15px;
    font-weight: 700;
    color: #f5f5f7;
    margin-bottom: 6px;
  }}
  .agy-flp-var-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 18px;
  }}
  @media (max-width: 1050px) {{
    .agy-flp-var-grid {{
      grid-template-columns: 1fr;
    }}
  }}
  .agy-flp-var-col {{
    background: #141418;
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 12px;
    padding: 14px 16px;
  }}
  .agy-flp-var-col-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #e4e4e7;
    padding-bottom: 8px;
    margin-bottom: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }}
  .agy-flp-var-chips {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }}
  .agy-flp-vchip {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 9px;
    border-radius: 7px;
    background: #1d1d22;
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #e4e4e7;
    font-family: var(--font-mono);
    font-size: 11.5px;
    cursor: help;
    transition: border-color 140ms ease, background 140ms ease, color 140ms ease;
  }}
  .agy-flp-vchip:hover {{
    border-color: #8052ff;
    background: rgba(128, 82, 255, 0.2);
    color: #ffffff;
  }}
  .agy-flp-vchip.calc {{
    border-color: rgba(255, 184, 41, 0.32);
    color: #fde68a;
  }}
  .agy-flp-vchip.const {{
    border-color: rgba(16, 185, 129, 0.32);
    color: #6ee7b7;
  }}
  .agy-flp-vchip.rejected {{
    border-color: rgba(239, 68, 68, 0.4);
    background: rgba(239, 68, 68, 0.12);
    color: #fca5a5;
  }}
  .agy-flp-vchip.opt {{
    border-color: rgba(16, 185, 129, 0.55);
    background: rgba(16, 185, 129, 0.16);
    color: #34d399;
    font-weight: 700;
  }}
  .agy-flp-mat-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-variant-numeric: tabular-nums;
    font-size: 13px;
  }}
  .agy-flp-mat-table th {{
    text-align: right;
    padding: 10px 12px;
    background: #18181d;
    color: #a1a1aa;
    font-family: var(--font-mono);
    font-size: 11.5px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    cursor: pointer;
    user-select: none;
  }}
  .agy-flp-mat-table th:first-child {{
    text-align: left;
    cursor: default;
  }}
  .agy-flp-mat-table th.col-opt {{
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border-bottom-color: rgba(16, 185, 129, 0.45);
  }}
  .agy-flp-mat-table th.col-focus {{
    outline: 1px solid #8052ff;
    color: #fff;
  }}
  .agy-flp-mat-table td {{
    padding: 9px 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    text-align: right;
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    color: #e4e4e7;
    transition: background 120ms ease;
  }}
  .agy-flp-mat-table td:first-child {{
    text-align: left;
    font-family: var(--font-sans);
  }}
  .agy-flp-mat-table td.cell-hoverable {{
    cursor: help;
  }}
  .agy-flp-mat-table td.cell-hoverable:hover {{
    background: rgba(128, 82, 255, 0.22) !important;
  }}
  .agy-flp-mat-table td.col-opt {{
    background: rgba(16, 185, 129, 0.08);
    color: #a7f3d0;
    font-weight: 700;
  }}
  .agy-flp-mat-table td.cell-warn {{
    color: #fbbf24;
    font-weight: 700;
  }}
  .agy-flp-mat-table td.cell-fail {{
    background: rgba(239, 68, 68, 0.14);
    color: #f87171;
    font-weight: 700;
  }}
  .agy-flp-mat-table td.cell-inf {{
    color: #71717a;
  }}
  .agy-flp-mat-table tr.mat-foot td {{
    background: #16161b;
    font-weight: 700;
    border-top: 1px solid rgba(255, 255, 255, 0.14);
    padding: 11px 12px;
  }}
  .agy-flp-chain {{
    display: flex;
    flex-wrap: wrap;
    align-items: stretch;
    gap: 8px;
    margin: 12px 0 16px 0;
  }}
  .agy-flp-node-step {{
    flex: 1 1 118px;
    background: #16161b;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 10px 11px;
    cursor: help;
    transition: border-color 140ms ease, transform 140ms ease;
  }}
  .agy-flp-node-step:hover {{
    border-color: #8052ff;
    transform: translateY(-2px);
  }}
  .agy-flp-node-step.step-cold {{
    border-color: rgba(16, 185, 129, 0.45);
    background: rgba(16, 185, 129, 0.08);
  }}
  .agy-flp-node-step.step-warn {{
    border-color: rgba(245, 158, 11, 0.45);
    background: rgba(245, 158, 11, 0.08);
  }}
  .agy-flp-node-step.step-danger {{
    border-color: rgba(239, 68, 68, 0.45);
    background: rgba(239, 68, 68, 0.08);
  }}
  .agy-flp-formula-card {{
    background: #141418;
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
  }}
  .agy-flp-formula-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }}
  .agy-flp-formula-title {{
    font-size: 13.5px;
    font-weight: 700;
    color: #f5f5f7;
  }}
  .agy-flp-formula-src {{
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: #a1a1aa;
  }}
  .agy-flp-esc-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
  }}
  .agy-flp-esc-table th {{
    text-align: left;
    padding: 10px 12px;
    background: #18181d;
    color: #a1a1aa;
    font-family: var(--font-mono);
    font-size: 11px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }}
  .agy-flp-esc-table td {{
    padding: 10px 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    color: #e4e4e7;
  }}
  .agy-flp-esc-table tr.esc-opt td {{
    background: rgba(16, 185, 129, 0.09);
    color: #ecfdf5;
    font-weight: 600;
  }}
  .agy-flp-esc-table tr.esc-hoverable {{
    cursor: help;
  }}
  .agy-flp-esc-table tr.esc-hoverable:hover td {{
    background: rgba(128, 82, 255, 0.16);
  }}
  .agy-tip-target {{
    cursor: help;
    border-radius: 6px;
    transition: background 120ms ease;
  }}
  .agy-tip-target:hover {{
    background: rgba(128, 82, 255, 0.12);
  }}
  /* Tooltip flotante técnico global (informe_calculo_centros_frio.md — explicación paso a paso) */
  #agy-flp-hover-tip {{
    position: fixed;
    top: 0;
    left: 0;
    z-index: 99999;
    width: 455px;
    max-width: min(475px, calc(100vw - 24px));
    max-height: calc(100vh - 28px);
    overflow-y: auto;
    background: #111116;
    color: #f5f5f7;
    border: 1px solid rgba(128, 82, 255, 0.52);
    border-radius: 12px;
    padding: 13px 15px;
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.72);
    pointer-events: none;
    opacity: 0;
    visibility: hidden;
    transform: translateY(4px);
    transition: opacity 110ms ease, transform 110ms ease;
    font-family: var(--font-sans);
    font-size: 12px;
    line-height: 1.48;
  }}
  #agy-flp-hover-tip.visible {{
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
  }}
  .agy-ht-steps {{
    margin: 4px 0 0 0;
    padding-left: 16px;
    display: flex;
    flex-direction: column;
    gap: 3px;
    color: #e4e4e7;
    font-size: 11.5px;
  }}
  .agy-ht-steps li {{
    line-height: 1.42;
  }}
  .agy-ht-decision {{
    margin-top: 5px;
    padding: 6px 8px;
    border-radius: 7px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.32);
    color: #a7f3d0;
    font-family: var(--font-mono);
    font-size: 11.2px;
    line-height: 1.4;
  }}
  .agy-ht-decision.warn {{
    background: rgba(239, 68, 68, 0.12);
    border-color: rgba(239, 68, 68, 0.35);
    color: #fecaca;
  }}
  .agy-ht-cat {{
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 9.8px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #c4b5fd;
    background: rgba(128, 82, 255, 0.18);
    border: 1px solid rgba(128, 82, 255, 0.35);
    padding: 2px 7px;
    border-radius: 999px;
    margin-bottom: 6px;
  }}
  .agy-ht-title {{
    font-family: var(--font-mono);
    font-size: 13.5px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 7px;
  }}
  .agy-ht-grid {{
    display: flex;
    flex-direction: column;
    gap: 5px;
  }}
  .agy-ht-row {{
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding-top: 4px;
    border-top: 1px solid rgba(255, 255, 255, 0.07);
  }}
  .agy-ht-k {{
    font-family: var(--font-mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #9ca3af;
  }}
  .agy-ht-v {{
    color: #e4e4e7;
    font-size: 11.8px;
    font-variant-numeric: tabular-nums;
  }}
  .agy-ht-v code {{
    font-family: var(--font-mono);
    font-size: 11px;
    color: #fde68a;
    background: rgba(255, 255, 255, 0.06);
    padding: 1px 4px;
    border-radius: 4px;
  }}
</style>

<!-- Etiqueta Serial Vertical estilo ORYZO AI -->
<div id="agy-oryzo-serial" aria-hidden="true">UNILLANOS · ING. SISTEMAS · 603301 · PUERTO GAITÁN, META · G=(V,E)</div>
<!-- Tooltip Técnico Flotante Global (Catálogo de Variables, Matriz 7×5, Simulación P05 y Escenarios) -->
<div id="agy-flp-hover-tip" role="tooltip" aria-hidden="true"></div>

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
      <span class="agy-tab-dot"></span>1. Problemáticas (Base)
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
      <span class="agy-tab-dot"></span>6. Programa y Teorías
    </button>
    <button class="agy-slide-tab is-editorial" role="tab" aria-selected="false" data-slide="6" onclick="agyCambiarSlide(6, false)">
      <span class="agy-tab-dot"></span>7. Cálculo FLP
    </button>
    <button class="agy-slide-tab is-editorial" role="tab" aria-selected="false" data-slide="7" onclick="agyCambiarSlide(7, false)">
      <span class="agy-tab-dot"></span>8. Comparación y Conclusión
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

    <!-- DIAPOSITIVA 1: PROBLEMÁTICA GENERAL, TODOS LOS PROBLEMAS CON DATOS REALES (LÍNEA BASE SIN SOLUCIÓN) Y SIMULADOR TÉRMICO -->
    <article id="agy-ed-slide-0" class="agy-ed-slide active">
      <div class="agy-hero-grid">
        <div>
          <span class="agy-badge" style="color:var(--color-saffron-spark);border-color:rgba(255,184,41,0.45);margin-bottom:12px;">FASE 1 · DIAGNÓSTICO Y LÍNEA BASE SIN SOLUCIÓN (ECA C05-01-2026-003)</span>
          <h1 class="agy-hero-title">
            Problemática Integral de la <em>Cadena de Frío</em> en la Pesca Artesanal de Puerto Gaitán, Meta.
          </h1>
          <p class="agy-hero-lead">
            En las cuencas de los ríos <strong>Meta, Manacacías e Iteviare</strong>, 7 asociaciones pesqueras extraen <strong>265.75 kg/día</strong> de pescado fresco a <strong>30 °C</strong> sin red frigorífica articulada, enfrentando hasta un <strong>40.0% de merma poscaptura</strong>, riesgo tóxico de <strong>histamina (&gt; 4.0 h FAO)</strong>, déficit total de hielo 1:1 y parálisis extractiva del 100% durante los 3 meses de veda (abril–junio).
          </p>
          <div class="agy-meta-authors">
            <div><strong>UNIVERSIDAD DE LOS LLANOS</strong> <span>· Fac. Ciencias Básicas e Ingeniería · Ing. en Sistemas</span></div>
            <div><strong>Autores:</strong> Yair Armando Ospino Herrera (160005565) · Harold Esteban Padilla Díaz (160005567) · Alejandro Ávila Ortiz (160005507) · Samuel Esteban Gómez Buitrago (160005533)</div>
            <div><strong>Curso:</strong> Estructuras de Datos (603301) · Corte 3: Grafos · Semestre 2026-II · Villavicencio, Meta</div>
          </div>
          <div class="agy-cta-row">
            <button class="agy-pill-cta" onclick="agyCambiarSlide(5, false)">
              2.º Ver Qué Hace el Programa y Algoritmos
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
            </button>
            <button class="agy-pill-ghost" onclick="agyCambiarSlide(4, false)">Explorar Grafo Mixto G=(V,E)</button>
            <button class="agy-pill-ghost" onclick="agyCambiarSlide(7, false)">Ir a Comparación Antes vs. Después</button>
          </div>
        </div>

        <!-- Constelación Interactiva en Vivo Dala (Partículas Triangulares + Nodos) -->
        <div class="agy-constellation-plinth">
          <canvas id="agy-hero-constellation" aria-label="Visualización de constelación de nodos logísticos"></canvas>
          <div class="agy-plinth-caption">
            <div>
              <div style="font-size:13px;font-weight:600;color:var(--color-warm-cream);">Red Territorial Auditada · Cuenca del Río Meta</div>
              <div style="font-family:var(--font-mono);font-size:12px;color:var(--color-ash-gray);">25 Vértices · 41 Aristas · 12,413 Vértices Fluviales IGAC · 49 Puntos de Campo</div>
            </div>
            <span class="agy-badge" style="color:var(--color-saffron-spark);border-color:rgba(255,184,41,0.4);">INTERACTIVO</span>
          </div>
        </div>
      </div>

      <!-- 4 Indicadores Maestros del Problema Base (Sin Solución — Interactivos al Clic) -->
      <div class="agy-bento-grid-4">
        <div class="agy-plinth-card highlight-saffron" role="button" tabindex="0" style="cursor:pointer;" onclick="agyAbrirModalDetalle('kpi_prod')" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();agyAbrirModalDetalle('kpi_prod');}}" title="Haz clic para ver la tabla completa de producción por asociación y fuentes AUNAP/ECA">
          <div class="agy-stat-num">265.8<span class="agy-stat-unit">kg/día</span></div>
          <div class="agy-stat-title">Producción Diaria Expuesta (7 Asociaciones)</div>
          <div class="agy-stat-desc">ASOPESGA (65.75 kg), Atarraya Verde (40 kg), Asovida (25 kg), Asopespor (20 kg), Pto. Trujillo (30 kg), Desembarcadero (50 kg) y Soplaviento (35 kg).</div>
          <span class="agy-modal-trigger" style="margin-top:10px;">+ Ver Tabla y Fuentes (AUNAP / ECA)</span>
        </div>
        <div class="agy-plinth-card" role="button" tabindex="0" style="cursor:pointer;" onclick="agyAbrirModalDetalle('kpi_merma')" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();agyAbrirModalDetalle('kpi_merma');}}" title="Haz clic para ver el desglose matemático y algorítmico del 40.0% de pérdida poscaptura">
          <div class="agy-stat-num" style="color:var(--color-ember-accent);">40.0<span class="agy-stat-unit">%</span></div>
          <div class="agy-stat-title">Pérdida Poscaptura Sin Cadena de Frío</div>
          <div class="agy-stat-desc">A 30 °C la cachama pierde 2.5%/h (y 3.75%/h en zonas D). Hacia Bogotá sin frío se pierde el 36.1% de la carga ($1,151,000 COP/día en merma).</div>
          <span class="agy-modal-trigger" style="margin-top:10px;">+ Ver 5 Modelos y Fórmulas de Merma</span>
        </div>
        <div class="agy-plinth-card highlight-iris" role="button" tabindex="0" style="cursor:pointer;" onclick="agyAbrirModalDetalle('kpi_fao')" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();agyAbrirModalDetalle('kpi_fao');}}" title="Haz clic para ver el fundamento biológico, Codex Alimentarius y algoritmo TCSPP del límite FAO">
          <div class="agy-stat-num">4.0<span class="agy-stat-unit">h (240 min)</span></div>
          <div class="agy-stat-title">Límite Sanitario FAO (Riesgo Histamina)</div>
          <div class="agy-stat-desc">Sin centros de frío, el viaje a mercado toma hasta 9.48 h (569 min = 237% sobre el límite FAO), disparando bacterias e histamina tóxica.</div>
          <span class="agy-modal-trigger" style="margin-top:10px;">+ Ver Fundamento FAO y Algoritmo TCSPP</span>
        </div>
        <div class="agy-plinth-card" role="button" tabindex="0" style="cursor:pointer;" onclick="agyAbrirModalDetalle('kpi_aislamiento')" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();agyAbrirModalDetalle('kpi_aislamiento');}}" title="Haz clic para ver el desglose tramo a tramo de los 63.0 km desde P05 Puerto Trujillo">
          <div class="agy-stat-num">63.0<span class="agy-stat-unit">km</span></div>
          <div class="agy-stat-title">Aislamiento Fluvial Crítico (P05 Iteviare)</div>
          <div class="agy-stat-desc">Puerto Trujillo navega 63 km (2.70 h) cruzando la zona de deterioro D01 y el muelle M02 (+15 min de espera al sol) sin energía eléctrica.</div>
          <span class="agy-modal-trigger" style="margin-top:10px;">+ Ver Desglose de Ruta (63.0 km / 2.95 h)</span>
        </div>
      </div>

      <!-- Catálogo Exhaustivo de las 6 Problemáticas Reales (Línea Base para Comparar Después + Modal Cards) -->
      <div class="agy-math-showcase" style="margin-bottom:28px;">
        <div class="agy-math-top">
          <div class="agy-math-heading">
            <h2>Las 6 Problemáticas Estructurales del Proyecto (Línea Base «Antes / Sin Solución»)</h2>
            <p>Haz clic en <strong>+ Ver Evidencia y Datos</strong> en cualquier tarjeta para abrir su ficha cuantitativa completa. Estos valores son la referencia exacta para el comparador final.</p>
          </div>
          <span class="agy-badge" style="color:var(--color-ember-accent);border-color:rgba(255,107,43,0.45);">DIAGNÓSTICO CUANTITATIVO REAL</span>
        </div>
        <div class="agy-bento-grid-3" style="margin-bottom:0;">
          <div class="agy-plinth-card obsidian">
            <div>
              <span class="agy-badge" style="color:var(--color-ember-accent);margin-bottom:8px;">PROBLEMA 1 · CINÉTICA TÉRMICA</span>
              <div class="agy-stat-title">Degradación Enzimática a 30 °C (2.5%/h a 3.75%/h)</div>
              <p style="font-size:13.5px;margin-top:6px;">Sin hielo, la vida útil total del pescado cae de <strong>360 h (15 días a 0 °C)</strong> a solo <strong>6.0 h (a 30 °C)</strong>. En envío local se pierden 6.3 kg/día (2.4%) y hacia Bogotá 45.4 a 95.9 kg/día (17.1%–36.1%).</p>
            </div>
            <button class="agy-modal-trigger" onclick="agyAbrirModalDetalle('prob1')">+ Ver Evidencia y Datos</button>
          </div>
          <div class="agy-plinth-card obsidian">
            <div>
              <span class="agy-badge" style="color:var(--color-ember-accent);margin-bottom:8px;">PROBLEMA 2 · INOCUIDAD Y SALUD</span>
              <div class="agy-stat-title">Generación de Histamina (&gt; 240 min sin frío)</div>
              <p style="font-size:13.5px;margin-top:6px;">Superar las <strong>4.0 h (240 min)</strong> sin abatimiento térmico produce descarboxilación bacteriana de histidina a <strong>histamina</strong> (toxina termoestable). Sin planta en Pto. Gaitán, el cumplimiento a Bogotá es <strong>0.0%</strong>.</p>
            </div>
            <button class="agy-modal-trigger" onclick="agyAbrirModalDetalle('prob2')">+ Ver Evidencia y Datos</button>
          </div>
          <div class="agy-plinth-card obsidian">
            <div>
              <span class="agy-badge" style="color:var(--color-saffron-spark);margin-bottom:8px;">PROBLEMA 3 · INFRAESTRUCTURA Y HIELO</span>
              <div class="agy-stat-title">Déficit Hielo 1:1 (265.8 kg/día) y Espera en Muelles</div>
              <p style="font-size:13.5px;margin-top:6px;">La norma FAO exige proporción <strong>1:1 hielo:pescado</strong> (265.75 kg hielo/día = $6.38M COP/mes) y erradicar cajas de madera. Los muelles M01 y M02 no tienen red eléctrica y agregan <strong>+0.25 h (+15 min)</strong> al sol.</p>
            </div>
            <button class="agy-modal-trigger" onclick="agyAbrirModalDetalle('prob3')">+ Ver Evidencia y Datos</button>
          </div>
          <div class="agy-plinth-card obsidian">
            <div>
              <span class="agy-badge" style="color:var(--color-electric-iris);margin-bottom:8px;">PROBLEMA 4 · COSTO DE INVERSIÓN FLP</span>
              <div class="agy-stat-title">Sobrecosto por Ubicación Empírica ($23.02M/mes)</div>
              <p style="font-size:13.5px;margin-top:6px;">Cada centro de frío cuesta <strong>$5,000,000 COP/mes</strong> de costo fijo. Abrir los 3 candidatos (F01+F02+F03) eleva el gasto a <strong>$23,023,272 COP/mes ($104,444/kg)</strong> usando apenas el 5.3% de su capacidad.</p>
            </div>
            <button class="agy-modal-trigger" onclick="agyAbrirModalDetalle('prob4')">+ Ver Evidencia y Datos</button>
          </div>
          <div class="agy-plinth-card obsidian">
            <div>
              <span class="agy-badge" style="color:var(--color-electric-link);margin-bottom:8px;">PROBLEMA 5 · ÉPOCA DE VEDA (ECA)</span>
              <div class="agy-stat-title">Parálisis Extractiva en Veda (Abril–Junio, 100%)</div>
              <p style="font-size:13.5px;margin-top:6px;">Durante 3 meses al año (25% del calendario), la veda AUNAP prohíbe el <strong>100% de la captura fluvial</strong> en los 7 puntos de río, dejando sin ingresos al pescador si no se conmuta a piscicultura nativa.</p>
            </div>
            <button class="agy-modal-trigger" onclick="agyAbrirModalDetalle('prob5')">+ Ver Evidencia y Datos</button>
          </div>
          <div class="agy-plinth-card obsidian">
            <div>
              <span class="agy-badge" style="color:#10b981;margin-bottom:8px;">PROBLEMA 6 · AMBIENTE Y TRAZABILIDAD</span>
              <div class="agy-stat-title">Presión en Hábitats H01/H02 y 0% Trazabilidad</div>
              <p style="font-size:13.5px;margin-top:6px;">El tránsito empírico invade el Humedal Manacacías (<strong>H01</strong>) y la Reserva Indígena Corocito (<strong>H02</strong>), mientras el <strong>100% de los lotes</strong> carece de registro digital de temperatura y tiempo ante ICA/INVIMA.</p>
            </div>
            <button class="agy-modal-trigger" onclick="agyAbrirModalDetalle('prob6')">+ Ver Evidencia y Datos</button>
          </div>
        </div>
      </div>

      <!-- ═══════════════════════════════════════════════════════════════════
           SIMULADOR TÉRMICO v2 — Estilo Highcharts (Canvas + Plot Bands + Tooltip)
           ═══════════════════════════════════════════════════════════════════ -->
      <div class="agy-crosshair-panel">
        <div class="agy-chart-header">
          <div>
            <h2>Simulador Interactivo de Degradación Térmica</h2>
            <p>Arrastra el cursor o pasa el ratón sobre el gráfico para comparar la degradación de <strong>265.75 kg/día</strong> de Cachama (<em>Colossoma macropomum</em>) a <strong>30 °C</strong> sin cadena de frío vs. con centro <strong>F02</strong> a 0 °C.</p>
          </div>
          <button type="button" id="agy-therm-formula-btn" class="agy-therm-formula-btn" onclick="agyToggleFormulaTermica()" aria-expanded="false" aria-controls="agy-therm-formula-drawer" title="Haz clic para ver el desglose matemático interactivo de cada variable">
            <div class="agy-therm-formula-main">
              <var onclick="event.stopPropagation();agyEnfocarVariableFormula('dacum')">D<sub>acum</sub>(t)</var>
              <span class="agy-op">=</span>
              <var onclick="event.stopPropagation();agyEnfocarVariableFormula('d0')">D<sub>0</sub></var>
              <span class="agy-op">+</span>
              <span class="agy-sum">&Sigma;</span>
              <span>(</span>
              <var onclick="event.stopPropagation();agyEnfocarVariableFormula('kbase')">k<sub>base</sub></var>
              <span class="agy-op">&middot;</span>
              <var onclick="event.stopPropagation();agyEnfocarVariableFormula('phid')">&phi;<sub><i>D</i></sub></var>
              <span class="agy-op">&middot;</span>
              <var onclick="event.stopPropagation();agyEnfocarVariableFormula('ti')">t<sub><i>i</i></sub></var>
              <span>)</span>
            </div>
            <span class="agy-therm-formula-hint">
              <span class="hint-dot"></span>
              <span id="agy-therm-formula-hint-txt">Haz clic en la fórmula o en sus términos para desglosar variables</span>
            </span>
          </button>
        </div>

        <!-- Panel Interactivo y Organizado de la Ecuación Cinética de Degradación -->
        <div id="agy-therm-formula-drawer" class="agy-formula-drawer" role="region" aria-label="Desglose de variables de la ecuación de degradación térmica">
          <div class="agy-formula-terms-bar">
            <button type="button" class="agy-term-pill active" data-term="all" onclick="agySeleccionarTerminoFormula('all')">
              Ver Todas las Variables (5)
            </button>
            <button type="button" class="agy-term-pill" data-term="dacum" onclick="agySeleccionarTerminoFormula('dacum')">
              <code>D<sub>acum</sub>(t)</code> Degradación Acumulada
            </button>
            <button type="button" class="agy-term-pill" data-term="d0" onclick="agySeleccionarTerminoFormula('d0')">
              <code>D<sub>0</sub></code> Deterioro Inicial
            </button>
            <button type="button" class="agy-term-pill" data-term="kbase" onclick="agySeleccionarTerminoFormula('kbase')">
              <code>k<sub>base</sub></code> Constante Cinética
            </button>
            <button type="button" class="agy-term-pill" data-term="phid" onclick="agySeleccionarTerminoFormula('phid')">
              <code>&phi;<sub><i>D</i></sub></code> Factor Zona D
            </button>
            <button type="button" class="agy-term-pill" data-term="ti" onclick="agySeleccionarTerminoFormula('ti')">
              <code>t<sub><i>i</i></sub></code> Tiempo en Tramo i
            </button>
          </div>

          <div class="agy-formula-grid" id="agy-formula-cards-grid">
            <!-- 1. D_acum(t) -->
            <div class="agy-formula-card highlighted" data-fcard="dacum" onclick="agySeleccionarTerminoFormula('dacum')">
              <div class="agy-fcard-head">
                <span class="agy-fcard-sym">D<sub>acum</sub>(t)</span>
                <span class="agy-fcard-title">Degradación Térmica Acumulada</span>
              </div>
              <div class="agy-fcard-body">
                Es el porcentaje total de pérdida de calidad/masa (<span class="agy-tex">0%</span> a <span class="agy-tex">100%</span>) que ha sufrido el lote de pescado tras transcurrir un tiempo total <span class="agy-tex"><var>t</var></span>.
                <div class="agy-fcard-mathbox">
                  <span class="agy-tex">0% <span class="agy-op">&le;</span> <var>D</var><sub>acum</sub>(<var>t</var>) <span class="agy-op">&le;</span> 100%</span>
                  &nbsp;&middot;&nbsp;
                  <span class="agy-tex"><var>t</var><sub>FAO</sub> <span class="agy-op">&le;</span> 4.0 h</span>
                </div>
              </div>
            </div>

            <!-- 2. D_0 -->
            <div class="agy-formula-card" data-fcard="d0" onclick="agySeleccionarTerminoFormula('d0')">
              <div class="agy-fcard-head">
                <span class="agy-fcard-sym">D<sub>0</sub></span>
                <span class="agy-fcard-title">Deterioro Inicial</span>
              </div>
              <div class="agy-fcard-body">
                Es el estado de frescura en el instante de la captura (<span class="agy-tex"><var>t</var> <span class="agy-op">=</span> 0</span>). En condiciones de pesca artesanal inmediata se asume:
                <div class="agy-fcard-mathbox">
                  <span class="agy-tex"><var>D</var><sub>0</sub> <span class="agy-op">=</span> 0%</span>
                </div>
              </div>
            </div>

            <!-- 3. k_base -->
            <div class="agy-formula-card" data-fcard="kbase" onclick="agySeleccionarTerminoFormula('kbase')">
              <div class="agy-fcard-head">
                <span class="agy-fcard-sym">k<sub>base</sub></span>
                <span class="agy-fcard-title">Constante Cinética de Degradación Bacteriana</span>
              </div>
              <div class="agy-fcard-body">
                Depende de la temperatura ambiente según las curvas de conservación de la FAO:
                <ul>
                  <li><strong>A temperatura ambiente de los Llanos (<span class="agy-tex">30 &deg;C</span>):</strong></li>
                </ul>
                <div class="agy-fcard-mathbox">
                  <span class="agy-tex"><var>k</var><sub>base</sub> <span class="agy-op">=</span> 2.5% / h <span class="agy-op">=</span> 0.025 h<sup>&minus;1</sup></span>
                </div>
                <ul>
                  <li><strong>Bajo refrigeración activa (<span class="agy-tex">0 &deg;C</span> en Centro F02 o hielo <span class="agy-tex">1:1</span>):</strong> El metabolismo bacteriano se frena en un 90%, cayendo a:</li>
                </ul>
                <div class="agy-fcard-mathbox">
                  <span class="agy-tex"><var>k</var><sub>frío</sub> <span class="agy-op">=</span> 0.25% / h <span class="agy-op">=</span> 0.0025 h<sup>&minus;1</sup></span>
                </div>
              </div>
            </div>

            <!-- 4. phi_D -->
            <div class="agy-formula-card" data-fcard="phid" onclick="agySeleccionarTerminoFormula('phid')">
              <div class="agy-fcard-head">
                <span class="agy-fcard-sym">&phi;<sub><i>D</i></sub></span>
                <span class="agy-fcard-title">Factor Multiplicador de Zona Crítica D</span>
              </div>
              <div class="agy-fcard-body">
                Modela la vulnerabilidad geográfica del tramo:
                <ul>
                  <li>En tramos fluviales o carreteras normales: <span class="agy-tex"><var>&phi;</var><sub><i>D</i></sub> <span class="agy-op">=</span> 1.0</span>.</li>
                  <li>En Zonas de Deterioro Térmico (<span class="agy-tex">D01, D02</span>): <span class="agy-tex"><var>&phi;</var><sub><i>D</i></sub> <span class="agy-op">=</span> 1.5</span>.</li>
                </ul>
                <div class="agy-fcard-mathbox">
                  <span class="agy-tex"><var>k</var><sub>zona D</sub> <span class="agy-op">=</span> 2.5% <span class="agy-op">&times;</span> 1.5 <span class="agy-op">=</span> 3.75% / h</span>
                </div>
              </div>
            </div>

            <!-- 5. t_i -->
            <div class="agy-formula-card" data-fcard="ti" onclick="agySeleccionarTerminoFormula('ti')">
              <div class="agy-fcard-head">
                <span class="agy-fcard-sym">t<sub><i>i</i></sub></span>
                <span class="agy-fcard-title">Tiempo en el Tramo i</span>
              </div>
              <div class="agy-fcard-body">
                Número de horas transcurridas en cada segmento dirigido <span class="agy-tex"><var>e</var><sub><i>i</i></sub> <span class="agy-op">&in;</span> <var>E</var></span> de la ruta logística.
                <div class="agy-fcard-mathbox">
                  <span class="agy-tex"><var>t</var> <span class="agy-op">=</span> <span class="agy-sum">&Sigma;</span> <var>t</var><sub><i>i</i></sub> <span class="agy-op">=</span> <strong id="agy-fcard-ti-live" style="color:#ffffff">2.20 h</strong></span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Leyenda Interactiva (toggle de series) -->
        <div class="agy-chart-legend" id="agy-therm-legend">
          <div class="agy-legend-item" data-series="sinFrio" onclick="agyToggleSerie('sinFrio', this)">
            <span class="agy-legend-swatch" style="background:#ff6b2b;"></span>Sin Solución (30 °C, 2.5%/h)
          </div>
          <div class="agy-legend-item" data-series="zonaD" onclick="agyToggleSerie('zonaD', this)">
            <span class="agy-legend-swatch" style="background:#dc5000;border:1px dashed #dc5000;background:transparent;"></span>Zona D (×1.5 → 3.75%/h)
          </div>
          <div class="agy-legend-item" data-series="conF02" onclick="agyToggleSerie('conF02', this)">
            <span class="agy-legend-swatch" style="background:#10b981;"></span>Con F02 (0 °C, 0.25%/h)
          </div>
        </div>

        <!-- Canvas del gráfico interactivo -->
        <div class="agy-chart-canvas-wrap" id="agy-therm-chart-wrap">
          <canvas id="agy-therm-canvas" height="320"></canvas>
          <div class="agy-chart-tooltip" id="agy-therm-tooltip"></div>
        </div>

        <!-- Controles: Slider + Presets -->
        <div class="agy-crosshair-controls">
          <div style="display:flex;align-items:center;gap:10px;flex:1;min-width:0;">
            <span style="font-family:var(--font-mono);font-size:12px;font-weight:700;color:var(--color-saffron-spark);white-space:nowrap;">t =</span>
            <input id="agy-sim-time-slider" class="agy-time-slider" type="range" min="0" max="10" step="0.05" value="2.2" oninput="agyActualizarCrosshairTermico(parseFloat(this.value))" aria-label="Horas de exposición térmica">
            <strong id="agy-sim-time-lbl" style="font-family:var(--font-mono);font-size:14px;color:var(--color-bone-white);white-space:nowrap;min-width:110px;">2.20 h (132 min)</strong>
          </div>
          <div class="agy-preset-btns">
            <button class="agy-preset-btn" onclick="agySetPreset(0.48, this)">P01 · 0.48 h</button>
            <button class="agy-preset-btn active-preset" onclick="agySetPreset(2.20, this)">P05+F02 · 2.20 h</button>
            <button class="agy-preset-btn" onclick="agySetPreset(4.00, this)">FAO · 4.0 h</button>
            <button class="agy-preset-btn" onclick="agySetPreset(9.48, this)">Bogotá · 9.48 h</button>
          </div>
        </div>

        <!-- Tarjetas de readout -->
        <div class="agy-sim-readout-row" id="agy-sim-readout-cards"></div>
      </div>
    </article>

    <!-- DIAPOSITIVA 6: QUÉ HACE EL PROGRAMA, TEORÍAS DE PROGRAMACIÓN, CONSTRUCCIÓN DE GRAFOS Y DEFENSA TÉCNICA -->
    <article id="agy-ed-slide-5" class="agy-ed-slide">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:16px;padding-bottom:24px;border-bottom:1px dashed var(--color-cork-border);margin-bottom:28px;">
        <div>
          <span class="agy-badge" style="color:var(--color-electric-iris);border-color:rgba(128,82,255,0.45);margin-bottom:10px;">FASE 2 · ARQUITECTURA DE SOFTWARE, TEORÍAS DE PROGRAMACIÓN Y CONSTRUCCIÓN DE GRAFOS</span>
          <h2>Qué Hace el Programa, Teorías de Programación y Construcción de los Grafos</h2>
          <p style="margin-top:6px;">Implementación 100% propia desde cero en Python — Explicación teórica, orden exacto de construcción (Vial, Ríos y Mixto), analogía del sistema y defensa cuantitativa.</p>
        </div>
        <div class="agy-cta-row">
          <button class="agy-pill-ghost" onclick="agyCambiarSlide(4, false)">Ver Grafo Mixto en Vivo</button>
          <button class="agy-pill-cta" onclick="agyCambiarSlide(6, false)">Ver Validación FLP →</button>
        </div>
      </div>

      <!-- BLOQUE 1: LAS 6 TEORÍAS DE PROGRAMACIÓN Y ESTRUCTURAS DE DATOS QUE USA EL PROGRAMA -->
      <div class="agy-math-showcase" style="margin-bottom:28px;">
        <div class="agy-math-top">
          <div class="agy-math-heading">
            <h3>1. Qué Hace el Programa (`main.py`) y Qué Teorías de Programación Utiliza</h3>
            <p>El software ejecuta un pipeline automatizado de 10 etapas apoyado en 6 paradigmas teóricos de Ciencias de la Computación. Haz clic en <strong>+ Qué es y Cómo se Aplica</strong> para ver su defensa técnica.</p>
          </div>
          <span class="agy-badge" style="color:#10b981;border-color:rgba(16,185,129,0.45);">0 LIBRERÍAS EXTERNAS DE CÁLCULO (ADR-01)</span>
        </div>

        <div class="agy-bento-grid-3" style="margin-bottom:0;">
          <div class="agy-plinth-card obsidian">
            <div>
              <span class="agy-badge" style="color:var(--color-saffron-spark);margin-bottom:8px;">TEORÍA 1 · GRAFOS Y TABLAS HASH O(1)</span>
              <div class="agy-stat-title">Lista de Adyacencia sobre Diccionarios Hash + POO</div>
              <p style="font-size:13.5px;margin-top:6px;"><strong>Qué es:</strong> Representa el grafo como un mapa disperso <code>_adyacencia: Dict[str, List[Arista]]</code> y <code>_nodos: Dict[str, Nodo]</code>.<br><strong>Por qué se usó:</strong> Con 25 nodos y 41 aristas (densidad 6.8%), evita desperdiciar el 93.2% de memoria de una matriz densa 25×25 y accede a cualquier vértice en <strong>O(1)</strong>.</p>
            </div>
            <button class="agy-modal-trigger" onclick="agyAbrirModalDetalle('teo1')">+ Qué es y Cómo se Defiende</button>
          </div>

          <div class="agy-plinth-card obsidian">
            <div>
              <span class="agy-badge" style="color:var(--color-electric-iris);margin-bottom:8px;">TEORÍA 2 · HEAP BINARIO Y PODA VORAZ</span>
              <div class="agy-stat-title">Dijkstra con Min-Heap (`heapq`) + Poda FAO (Branch &amp; Bound)</div>
              <p style="font-size:13.5px;margin-top:6px;"><strong>Qué es:</strong> Estrategia voraz (<em>Greedy</em>) sobre árbol binario completo en arreglo donde el padre siempre es menor o igual a sus hijos.<br><strong>Aplicación:</strong> Extrae el nodo de menor costo/tiempo en <strong>O(log V)</strong> y poda en vivo toda ruta cuyo tiempo sin frío supere <strong>4.0 h (240 min)</strong>.</p>
            </div>
            <button class="agy-modal-trigger" onclick="agyAbrirModalDetalle('teo2')">+ Qué es y Cómo se Defiende</button>
          </div>

          <div class="agy-plinth-card obsidian">
            <div>
              <span class="agy-badge" style="color:var(--color-electric-link);margin-bottom:8px;">TEORÍA 3 · PROGRAMACIÓN DINÁMICA</span>
              <div class="agy-stat-title">Bellman-Ford O(V·E) y Floyd-Warshall O(V³)</div>
              <p style="font-size:13.5px;margin-top:6px;"><strong>Qué es:</strong> Resuelve problemas complejos descomponiéndolos en subproblemas superpuestos con principio de optimalidad.<br><strong>Aplicación:</strong> Bellman-Ford relaja aristas en V−1 fases y certifica <strong>0 ciclos negativos</strong>; Floyd-Warshall calcula la matriz 25×25 todos-contra-todos.</p>
            </div>
            <button class="agy-modal-trigger" onclick="agyAbrirModalDetalle('teo3')">+ Qué es y Cómo se Defiende</button>
          </div>

          <div class="agy-plinth-card obsidian">
            <div>
              <span class="agy-badge" style="color:#10b981;margin-bottom:8px;">TEORÍA 4 · BÚSQUEDA EN GRAFOS (BFS / DFS)</span>
              <div class="agy-stat-title">Cola FIFO (`deque`) en BFS y Pila con Backtracking en DFS</div>
              <p style="font-size:13.5px;margin-top:6px;"><strong>Qué es:</strong> Exploración sistemática por niveles (BFS en O(V+E)) y en profundidad con retroceso (DFS).<br><strong>Aplicación:</strong> BFS halla la ruta con <strong>menor número de transbordos</strong> hacia F02; DFS enumera todas las rutas simples de contingencia si se bloquea una vía.</p>
            </div>
            <button class="agy-modal-trigger" onclick="agyAbrirModalDetalle('teo4')">+ Qué es y Cómo se Defiende</button>
          </div>

          <div class="agy-plinth-card obsidian">
            <div>
              <span class="agy-badge" style="color:var(--color-saffron-spark);margin-bottom:8px;">TEORÍA 5 · DAG TOPOLÓGICO Y GRAFOS G(t)</span>
              <div class="agy-stat-title">Orden Topológico de Kahn O(V+E) y Grafo Bitemporal</div>
              <p style="font-size:13.5px;margin-top:6px;"><strong>Qué es:</strong> Un DAG es un grafo dirigido sin ciclos; el algoritmo de Kahn ordena dependencias por grado de entrada 0.<br><strong>Aplicación:</strong> Certifica la trazabilidad sanitaria ICA/INVIMA de los 7 lotes y conmuta la topología entre pesca (G_pesca) y veda (G_veda).</p>
            </div>
            <button class="agy-modal-trigger" onclick="agyAbrirModalDetalle('teo5')">+ Qué es y Cómo se Defiende</button>
          </div>

          <div class="agy-plinth-card obsidian">
            <div>
              <span class="agy-badge" style="color:var(--color-ember-accent);margin-bottom:8px;">TEORÍA 6 · OPTIMIZACIÓN COMBINATORIA</span>
              <div class="agy-stat-title">Facility Location Problem (FLP Greedy + MILP)</div>
              <p style="font-size:13.5px;margin-top:6px;"><strong>Qué es:</strong> Selección óptima de instalaciones sujetas a capacidad, cobertura temporal y costo fijo/variable.<br><strong>Aplicación:</strong> Evalúa los subconjuntos de <code>F01, F02, F03</code> y demuestra que abrir <strong>Solo F02</strong> minimiza el costo en $13.02M/mes con 100% de cobertura FAO.</p>
            </div>
            <button class="agy-modal-trigger" onclick="agyAbrirModalDetalle('teo6')">+ Qué es y Cómo se Defiende</button>
          </div>
        </div>
      </div>

      <!-- BLOQUE 2: METODOLOGÍA TEÓRICA Y PRÁCTICA DE CONSTRUCCIÓN DE LOS GRAFOS + ANALOGÍA Y DEFENSA -->
      <div class="agy-math-showcase" style="margin-bottom:28px;">
        <div class="agy-math-top">
          <div class="agy-math-heading">
            <h3>2. Cómo se Construyeron los Grafos (Vial, Ríos y Unión Multimodal), Analogía y Por Qué se Decidió Así</h3>
            <p>Selecciona cada etapa para ver exactamente qué información se usó, qué se creó primero en memoria, cómo se unieron ambos grafos y cómo defender cada decisión de diseño.</p>
          </div>
          <span class="agy-badge" style="color:var(--color-saffron-spark);border-color:rgba(255,184,41,0.45);">GUÍA DE DEFENSA Y ARQUITECTURA</span>
        </div>

        <div class="agy-step-tabs" role="tablist">
          <button class="agy-step-tab active" onclick="agyCambiarPasoConstruccion(0)" id="agy-step-btn-0">Paso 1 · Estructura Base y 7 Tipos de Vértices</button>
          <button class="agy-step-tab" onclick="agyCambiarPasoConstruccion(1)" id="agy-step-btn-1">Paso 2 · Construcción del Grafo Vial (OSRM)</button>
          <button class="agy-step-tab" onclick="agyCambiarPasoConstruccion(2)" id="agy-step-btn-2">Paso 3 · Construcción del Grafo de Ríos (rios.shp)</button>
          <button class="agy-step-tab" onclick="agyCambiarPasoConstruccion(3)" id="agy-step-btn-3">Paso 4 · Unión Multimodal, Analogía y Defensa</button>
        </div>

        <!-- PANE 0: ESTRUCTURA BASE Y VÉRTICES -->
        <div class="agy-step-pane active" id="agy-step-pane-0">
          <div class="agy-bento-grid-2" style="margin-bottom:0;">
            <div class="agy-plinth-card obsidian">
              <h3>Qué se hizo exactamente primero en código (`main.py` → `cargar_datos()`)</h3>
              <p style="font-size:14px;">1.º <strong>Primero se instanció el contenedor vacío</strong> <code>grafo = GrafoLogistico()</code> en memoria con dos diccionarios Hash vacíos: <code>_nodos = {{}}</code> y <code>_adyacencia = {{}}</code>.<br>
              2.º <strong>Después se cargaron y tipificaron los 25 vértices</strong> desde 3 archivos CSV verificables (<code>nodos_pesca.csv</code>, <code>nodos_centros.csv</code> y <code>nodos_destinos.csv</code>) asignando a cada objeto <code>Nodo</code> su categoría semántica (<code>P, M, A, F, D, H, V</code>), coordenadas WGS84 y reglas operativas.<br>
              3.º <strong>Finalmente se inyectaron las 41 aristas dirigidas</strong> desde <code>aristas.csv</code> con 19 campos reales y 3 tipos de vía (<code>fluvial</code>, <code>acceso_local</code>, <code>terrestre</code>).</p>
            </div>
            <div class="agy-plinth-card obsidian">
              <h3>Por qué 7 Tipos de Vértices Reales y no Nodos Iguales (`ADR-04`)</h3>
              <p style="font-size:14px;">En un grafo académico simple todos los vértices son iguales; en la realidad de Puerto Gaitán eso daría resultados falsos. Por eso cada tipo altera matemáticamente el flujo:<br>
              • <strong>P (7 Pesca):</strong> Inyectan los 265.75 kg/día.<br>
              • <strong>M (3 Muelles):</strong> Transbordo agua→tierra (+0.25 h de espera al sol sin frío).<br>
              • <strong>A (3 Acopios):</strong> Consolidación obligatoria antes del frío (<code>ADR-06</code>).<br>
              • <strong>F (3 Centros Frío):</strong> Abaten a 0 °C, reinician <code>t_sin_frio = 0</code> (capacidad 5,000 kg).<br>
              • <strong>D (2 Deterioro):</strong> Multiplican la pérdida térmica del tramo por <strong>×1.5</strong> (3.75%/h).<br>
              • <strong>H (2 Hábitats/Resguardos):</strong> Con <code>evitar_restringidos=True</code> quedan bloqueados como paso intermedio.<br>
              • <strong>V (5 Mercados):</strong> Nodos destino de demanda comercial.</p>
            </div>
          </div>
        </div>

        <!-- PANE 1: GRAFO VIAL -->
        <div class="agy-step-pane" id="agy-step-pane-1">
          <div class="agy-bento-grid-2" style="margin-bottom:0;">
            <div class="agy-plinth-card obsidian">
              <h3>Construcción Teórica y Práctica del Subgrafo Vial (`31 Aristas`, `19 Vértices`)</h3>
              <p style="font-size:14px;">• <strong>Fuentes de información:</strong> Levantamiento de vías terrestres entre muelles, acopios, centros de frío y mercados + API pública de ruteo <strong>OSRM (Open Source Routing Machine)</strong> en <code>src/visualization/route_geometry.py</code>.<br>
              • <strong>Clasificación de carreteras:</strong> Se separaron en <strong>12 aristas de <code>acceso_local</code></strong> (trochas y caminos veredales en estado regular/malo que conectan muelles y acopios con velocidades menores y mayor riesgo de retraso) y <strong>19 aristas <code>terrestres</code></strong> (corredor pavimentado Ruta Nacional 40 Puerto Gaitán → Puerto López → Villavicencio → Bogotá).<br>
              • <strong>Anclaje geométrico y caché:</strong> Para cada par <code>(origen, destino)</code> vial, <code>_anclar_extremos()</code> une la polilínea real OSRM con la coordenada exacta del vértice y la guarda en <code>data/output/rutas_cache.json</code>.</p>
            </div>
            <div class="agy-plinth-card obsidian">
              <h3>Defensa Técnica: ¿Por qué OSRM con Caché Persistente y no Distancias Rectas?</h3>
              <p style="font-size:14px;">1. <strong>Fidelidad topográfica:</strong> Una línea recta entre Puerto Gaitán y Bogotá mide ~215 km, pero por la carretera real (cruzando el piedemonte llanero en Villavicencio y Boquerón) son <strong>289 a 295.5 km</strong> con peajes y pendientes.<br>
              2. <strong>Autonomía 100% offline y cero costo (`ADR-03`):</strong> Al persistir las 48 polilíneas en <code>rutas_cache.json</code>, el programa corre en <strong>menos de 2 segundos sin internet</strong> y sin pagar licencias comerciales (como Google Maps API o Felt de pago).</p>
            </div>
          </div>
        </div>

        <!-- PANE 2: GRAFO DE RÍOS -->
        <div class="agy-step-pane" id="agy-step-pane-2">
          <div class="agy-bento-grid-2" style="margin-bottom:0;">
            <div class="agy-plinth-card obsidian">
              <h3>Construcción del Grafo Hidrológico Fluvial (`RedFluvial` sobre `rios.shp`)</h3>
              <p style="font-size:14px;">• <strong>Fuente cartográfica:</strong> Shapefile oficial del IGAC <code>data/shapefiles/rios.shp</code> en proyección <code>EPSG:4326</code> recortado al <em>bounding box</em> de la cuenca.<br>
              • <strong>Discretización en Grafo No Dirigido:</strong> En <code>src/visualization/river_network.py</code>, cada geometría <code>LineString</code> se descompone en <strong>12,413 vértices <code>(lon, lat)</code></strong> (redondeados a 5 decimales ≈ 1.1 m para fusionar confluencias de ríos) y <strong>24,790 aristas fluviales</strong> ponderadas con la fórmula geodésica de <strong>Haversine</strong>.<br>
              • <strong>Índice Espacial Hash (Rejilla de 447 celdas de 0.05°×0.05°):</strong> Permite proyectar cualquier punto de pesca al cauce navegable más cercano (<code>enlace_al_cauce()</code>) en tiempo constante sin recorrer los 12,413 vértices.<br>
              • <strong>Dijkstra Fluvial Propio:</strong> Calcula sobre los meandros reales del río las <strong>10 aristas fluviales</strong> del modelo.</p>
            </div>
            <div class="agy-plinth-card obsidian">
              <h3>Defensa Técnica: ¿Por qué un Segundo Grafo Fluvial No Dirigido?</h3>
              <p style="font-size:14px;">1. <strong>Por qué no dirigido en el agua y dirigido en la logística:</strong> Las canoas a motor navegan físicamente aguas arriba y aguas abajo sobre el mismo cauce (por eso <code>RedFluvial</code> es no dirigido), pero el flujo comercial del pescado va estrictamente de captura a mercado (por eso <code>GrafoLogistico</code> es dirigido).<br>
              2. <strong>Por qué desacoplar los 12,413 vértices del grafo de 25 nodos:</strong> Si metiéramos los 12,413 vértices del shapefile dentro de Floyd-Warshall <span class="agy-tex"><var>O</var>(<var>V</var><sup>3</sup>)</span>, haría <span class="agy-tex">1.9 <span class="agy-op">&times;</span> 10<sup>12</sup></span> operaciones (tardaría horas). Al precalcular los 10 tramos fluviales reales sobre <code>RedFluvial</code> e inyectarlos como 10 aristas ponderadas en <code>GrafoLogistico</code>, Floyd-Warshall tarda <strong>menos de 1 milisegundo</strong> (<span class="agy-tex">25<sup>3</sup> <span class="agy-op">=</span> 15,625</span> pasos) conservando la geometría real del río.</p>
            </div>
          </div>
        </div>

        <!-- PANE 3: UNIÓN Y ANALOGÍA -->
        <div class="agy-step-pane" id="agy-step-pane-3">
          <div class="agy-bento-grid-2" style="margin-bottom:0;">
            <div class="agy-plinth-card obsidian">
              <h3>La Analogía de Ingeniería y Cómo se Unen Ambos Grafos en <span class="agy-tex"><var>G</var> <span class="agy-op">=</span> (<var>V</var>, <var>E</var>)</span></h3>
              <p style="font-size:14px;">• <strong>Punto de soldadura (Los Nodos Puente `M` y `A`):</strong> El Grafo de Ríos (10 aristas azules) y el Grafo Vial (31 aristas ámbar/violeta) se unen exactamente en los <strong>Muelles de Transbordo (<code>M01, M02, M03</code>) y Acopios Ribereños (<code>A01, A02, A03</code>)</strong>, que tienen grado de entrada fluvial <span class="agy-tex"><span class="agy-op">&gt;</span> 0</span> y grado de salida terrestre <span class="agy-tex"><span class="agy-op">&gt;</span> 0</span>.<br>
              • <strong>Analogía de una Red de Transmisión con Regeneradores de Señal:</strong><br>
              — <em>Los ríos y canoas</em> son líneas de baja velocidad donde la "señal" (frescura del pescado) sufre atenuación térmica continua (<span class="agy-tex">2.5% / h</span>).<br>
              — <em>Los Muelles (M)</em> son convertidores de medio físico (agua → camión) con una latencia fija de <span class="agy-tex">+15 min</span>.<br>
              — <em>Los Centros de Frío (F)</em> actúan como <strong>repetidores regenerativos</strong>: antes de que la señal caiga del umbral de <span class="agy-tex">4.0 h</span> (histamina), abaten la temperatura a <span class="agy-tex">0 &deg;C</span> y restauran la vida útil a <span class="agy-tex">360 h</span>.<br>
              — <em>Los Hábitats (H)</em> son cortafuegos (<em>firewalls</em>) que bloquean el enrutamiento por zonas prohibidas.</p>
            </div>
            <div class="agy-plinth-card obsidian">
              <h3>Criterio de Defensa: Preguntas Clave que Puedes Responder con Total Seguridad</h3>
              <p style="font-size:14px;">• <strong>¿Por qué usar un peso multicriterio de 6 factores (<span class="agy-tex"><var>w</var><sub><i>ij</i></sub></span>) y no solo tiempo o distancia?</strong> Porque la ruta más corta en km puede ser una trocha destapada con 35% de probabilidad de retraso y alto riesgo térmico; el peso combinado equilibra Tiempo (25%), Costo (20%), Riesgo (20%), Pérdida (10%), Inestabilidad (10%) y Retraso (5%).<br>
              • <strong>¿Por qué implementar simultáneamente Dijkstra, Bellman-Ford y Floyd-Warshall?</strong> Porque cumplen roles complementarios: <strong>Dijkstra</strong> resuelve el ruteo puntual en <span class="agy-tex"><var>O</var>((<var>V</var> + <var>E</var>) log <var>V</var>)</span>, <strong>Bellman-Ford</strong> audita formalmente la ausencia de ciclos negativos y valida al 100% los tiempos de Dijkstra, y <strong>Floyd-Warshall</strong> entrega la matriz completa <span class="agy-tex">25 <span class="agy-op">&times;</span> 25</span> que necesita el optimizador FLP para evaluar cualquier combinación de centros en <span class="agy-tex"><var>O</var>(1)</span>.</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Simulador Interactivo de la Función Objetivo Multicriterio (ADR-02) -->
      <div class="agy-math-showcase">
        <div class="agy-math-top" style="border-bottom:none;padding-bottom:0;">
          <div class="agy-math-heading">
            <h3>3. Auditor Interactivo de la Función Multicriterio por Arista (ADR-02)</h3>
            <p>Selecciona cualquiera de las 41 aristas reales para inspeccionar cómo el programa calcula su peso combinado.</p>
          </div>
          <div class="agy-math-eq">
            <var>w</var><sub><i>ij</i></sub> <span class="agy-op">=</span> 0.25<var>t&#770;</var><sub><i>ij</i></sub> <span class="agy-op">+</span> 0.20<var>c&#770;</var><sub><i>ij</i></sub> <span class="agy-op">+</span> 0.20<var>r</var><sub>comb</sub> <span class="agy-op">+</span> 0.10<var>p&#770;</var><sub><i>ij</i></sub> <span class="agy-op">+</span> 0.10(1<span class="agy-op">&minus;</span><var>q</var><sub><i>ij</i></sub>) <span class="agy-op">+</span> 0.05<var>P</var><sub>ret</sub>
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

    <!-- DIAPOSITIVA 7: VISTA DEDICADA DE CÁLCULO FLP (PASO 2 & PASO 3: ORIGEN, VARIABLES, MATRIZ 7×5, FÓRMULAS Y ESCENARIOS) -->
    <article id="agy-ed-slide-6" class="agy-ed-slide">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:16px;padding-bottom:20px;border-bottom:1px dashed var(--color-cork-border);margin-bottom:22px;">
        <div>
          <span class="agy-badge" style="color:var(--color-saffron-spark);border-color:rgba(255,184,41,0.45);margin-bottom:10px;">FASE 3 · AUDITORÍA COMPLETA DEL CÁLCULO DE CENTROS DE FRÍO (FLP)</span>
          <h2>Cálculo de Centros de Frío: Origen, Variables, Matriz 7×5 y Fórmulas</h2>
          <p style="margin-top:6px;">Flujo de demostración paso a paso. <strong style="color:#c4b5fd;">Pasa el cursor</strong> sobre cualquier variable de nodo/arista, celda de la matriz, paso térmico o escenario para ver su archivo fuente, tipo y fórmula exacta.</p>
        </div>
        <div class="agy-cta-row">
          <button type="button" class="agy-pill-ghost" onclick="agyCambiarSlide(1, false)">← Volver al Mapa SIG</button>
          <div class="agy-ctrl-group">
            <button id="agy-ed-dest-pg" class="agy-ctrl-btn active" onclick="agyCambiarDestinoEditorial('PG')">Local (V_PG1)</button>
            <button id="agy-ed-dest-bog" class="agy-ctrl-btn" onclick="agyCambiarDestinoEditorial('BOG')">★ Bogotá (V_BOG1)</button>
          </div>
          <button class="agy-pill-cta" onclick="agyCambiarSlide(7, false)">Ir a Comparación Antes vs. Después →</button>
        </div>
      </div>

      <!-- 4 KPIs Verificados de la Solución Propuesta (con hover técnico) -->
      <div class="agy-bento-grid-4" style="margin-bottom:18px;">
        <div class="agy-plinth-card highlight-saffron agy-tip-target" data-flp-tip="kpi_solo_f02">
          <div class="agy-stat-num">Solo F02</div>
          <div class="agy-stat-title">Configuración Óptima FLP</div>
          <div class="agy-stat-desc">Puerto Gaitán urbano (<code>4.3128, -72.0826</code>) consolida los 265.75 kg/día usando solo el 5.3% de sus 5,000 kg.</div>
        </div>
        <div class="agy-plinth-card highlight-iris agy-tip-target" data-flp-tip="kpi_cobertura_fao">
          <div class="agy-stat-num">100<span class="agy-stat-unit">%</span></div>
          <div class="agy-stat-title">Cobertura Sanitaria FAO (7/7)</div>
          <div class="agy-stat-desc">Dijkstra FLP (<code>tiempo_h</code>): 7/7 en ≤ 2.65 h (P05). Con poda <code>dijkstra_con_cota_inocuidad</code>: máx. 2.10 h.</div>
        </div>
        <div class="agy-plinth-card agy-tip-target" data-flp-tip="kpi_costo_opt">
          <div class="agy-stat-num">$13.02<span class="agy-stat-unit">M/mes</span></div>
          <div class="agy-stat-title">Costo Operativo Optimizado</div>
          <div class="agy-stat-desc">$13,023,272 COP/mes hacia Bogotá ($59,080/kg) y $5,523,133 COP/mes en distribución local ($21,285/kg).</div>
        </div>
        <div class="agy-plinth-card agy-tip-target" data-flp-tip="kpi_ahorro_flp">
          <div class="agy-stat-num" style="color:#10b981;">−$10.0<span class="agy-stat-unit">M/mes</span></div>
          <div class="agy-stat-title">Ahorro Directo Mensual (−43.4%)</div>
          <div class="agy-stat-desc">Evita operar 3 plantas simultáneas ($23,023,272 COP/mes) sin sacrificar cobertura ni aumentar la merma.</div>
        </div>
      </div>

      <!-- Barra de 4 Sub-vistas Dedicadas (Paso 2 y Paso 3 de la Propuesta Visual) -->
      <div class="agy-flp-subtabs" role="tablist" aria-label="Pasos del cálculo FLP">
        <button id="agy-flp-tab-0" class="agy-flp-subtab active" role="tab" onclick="agyCambiarSubtabFLP(0)">1. Origen, Filtro y Variables</button>
        <button id="agy-flp-tab-1" class="agy-flp-subtab" role="tab" onclick="agyCambiarSubtabFLP(1)">2. Matriz Dijkstra 7×5 y Greedy/MILP</button>
        <button id="agy-flp-tab-2" class="agy-flp-subtab" role="tab" onclick="agyCambiarSubtabFLP(2)">3. Fórmulas y Simulación Térmica P05</button>
        <button id="agy-flp-tab-3" class="agy-flp-subtab" role="tab" onclick="agyCambiarSubtabFLP(3)">4. 5 Escenarios, Defensa y Auditoría</button>
      </div>

      <!-- SUBTAB 0: ORIGEN DE CANDIDATOS, FILTRO BOOLEANO Y CATÁLOGO INTERACTIVO DE VARIABLES -->
      <section id="agy-flp-pane-0" class="agy-flp-pane active">
        <div id="agy-flp-origin-wrap"></div>
      </section>

      <!-- SUBTAB 1: MATRIZ DIJKSTRA 7×5 (P01..P07 × 5 CANDIDATOS) + GREEDY/MILP -->
      <section id="agy-flp-pane-1" class="agy-flp-pane">
        <div id="agy-flp-matrix-wrap"></div>
      </section>

      <!-- SUBTAB 2: FÓRMULAS APLICADAS + SIMULACIÓN TÉRMICA RUTA CRÍTICA P05 -->
      <section id="agy-flp-pane-2" class="agy-flp-pane">
        <div id="agy-flp-thermal-wrap"></div>
        <div class="agy-plinth-card" style="margin-top:16px;">
          <h3>Tiempo Acumulado Sin Frío por Asociación vs. Límite FAO (240 min)</h3>
          <p style="font-size:13.5px;">Haz clic sobre cualquier asociación para abrir su itinerario óptimo directamente en el Grafo Multimodal.</p>
          <div id="agy-ed-thermal-chart" style="margin-top:10px;"></div>
        </div>
      </section>

      <!-- SUBTAB 3: EVALUACIÓN DE LOS 5 ESCENARIOS + CRITERIO DE DEFENSA Y AUDITORÍA -->
      <section id="agy-flp-pane-3" class="agy-flp-pane">
        <div id="agy-flp-scenarios-wrap"></div>

        <div class="agy-plinth-card" style="margin:16px 0;">
          <h3>Gráfico Comparativo de Costo Mensual y Pérdida por Escenario FLP</h3>
          <p style="font-size:13.5px;">Evaluación técnica de los 5 escenarios reales mediante <code>EvaluadorEscenarios</code> y <code>SimuladorCadenaFrio</code>.</p>
          <div id="agy-ed-flp-chart" style="margin-top:10px;"></div>
        </div>

        <!-- Guía de Defensa y Cuestionamiento Crítico (Cifras 100% Verificadas) -->
        <div class="agy-math-showcase" style="margin-bottom:16px;">
          <div class="agy-math-top">
            <div class="agy-math-heading">
              <h3>Criterio de Defensa: Cómo Defender Numéricamente la Decisión «Solo F02»</h3>
              <p>Respuestas cuantitativas exactas verificadas contra el código y los CSV ante objeciones del jurado.</p>
            </div>
            <span class="agy-badge" style="color:#10b981;border-color:rgba(16,185,129,0.45);">DEFENSA AUDITADA</span>
          </div>
          <div class="agy-bento-grid-3" style="margin-bottom:0;">
            <div class="agy-plinth-card obsidian">
              <div class="agy-stat-title" style="color:var(--color-saffron-spark);">1. ¿Por qué F02 y no F01 (Alto Neblinas)?</div>
              <p style="font-size:13.2px;">Las asociaciones del sur/Iteviare (<strong>P03 + P04 + P05 = 75.0 kg/día, 28.2% de la oferta</strong>; 25 + 20 + 30 kg/día) entran por A01 y M02 hacia el casco urbano y en el grafo dirigido <strong>no alcanzan F01 (∞)</strong>. Así F01 solo cubre <strong>4/7 (57.1%)</strong>, mientras <strong>F02</strong> cubre <strong>7/7 (100%)</strong> con energía trifásica y salida directa a la Ruta 40.</p>
            </div>
            <div class="agy-plinth-card obsidian">
              <div class="agy-stat-title" style="color:var(--color-electric-iris);">2. ¿Por qué no F03 (Puerto López)?</div>
              <p style="font-size:13.2px;">Puerto López (F03) está a <strong>126 km al occidente</strong> por la Ruta Nacional 40. Sin frío previo en Puerto Gaitán, <strong>P05 → F03 tarda 5.15 h</strong> (violando el límite FAO de 4.0 h). Además, el grafo dirigido tiene <code>F02→F03</code> sin retorno a <code>V_PG1</code>, por lo que su cobertura local es 0%.</p>
            </div>
            <div class="agy-plinth-card obsidian">
              <div class="agy-stat-title" style="color:#10b981;">3. ¿Por qué no abrir los 3 centros (F01+F02+F03)?</div>
              <p style="font-size:13.2px;">La oferta total censada (<strong>265.75 kg/día</strong>) ocupa apenas el <strong>5.3%</strong> de un solo centro (<strong>5,000 kg</strong>). Abrir 3 plantas eleva el costo fijo de <strong>$5M a $15M COP/mes (+200%)</strong> mientras la merma permanece igual (2.4% local / 17.1% Bogotá): pagar <strong>$10M COP/mes extra por 0.0 kg de mejora</strong>.</p>
            </div>
          </div>
        </div>

        <!-- Notas Críticas de Auditoría (Sección 5 de informe_calculo_centros_frio.md) -->
        <div id="agy-flp-audit-wrap"></div>
      </section>
    </article>

    <!-- DIAPOSITIVA 8: COMPARADOR INTERACTIVO ANTES VS DESPUÉS Y DELIMITACIÓN DE ÁREAS EXTERNAS -->
    <article id="agy-ed-slide-7" class="agy-ed-slide">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:16px;padding-bottom:24px;border-bottom:1px dashed var(--color-cork-border);margin-bottom:28px;">
        <div>
          <span class="agy-badge" style="color:#10b981;border-color:rgba(16,185,129,0.45);margin-bottom:10px;">FASE 4 · COMPARACIÓN INTERACTIVA ANTES VS. DESPUÉS Y DELIMITACIÓN DE ALCANCE</span>
          <h2>Comparación Interactiva (Sin Solución vs. Con Solución) y Conclusiones</h2>
          <p style="margin-top:6px;">Contraste numérico y didáctico de todas las problemáticas iniciales frente al modelo optimizado G=(V,E), y separación de problemas externos.</p>
        </div>
        <div class="agy-cta-row">
          <button class="agy-pill-cta" onclick="agyCambiarSlide(4, false)">Ir al Grafo Mixto Interactivo</button>
          <button class="agy-pill-ghost" onclick="agyCambiarSlide(0, false)">Volver a Problemáticas (Slide 1)</button>
        </div>
      </div>

      <!-- LABORATORIO COMPARADOR INTERACTIVO: ANTES (SIN SOLUCIÓN) VS DESPUÉS (CON SOLUCIÓN) -->
      <div class="agy-math-showcase" style="margin-bottom:28px;">
        <div class="agy-math-top">
          <div class="agy-math-heading">
            <h3>1. Comparador Interactivo: Problemáticas Sin Solución (Antes) vs. Con Solución Aplicada (Después)</h3>
            <p>Filtra por dimensión o selecciona una asociación pesquera específica para comparar interactivamente las métricas antes y después del programa.</p>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;">
            <select id="agy-comp-filter" class="agy-ctrl-select" onchange="agyRenderizarComparadorInteractivo()" aria-label="Filtrar problemática a comparar">
              <option value="all">Ver las 6 Problemáticas Comparadas</option>
              <option value="merma">1. Pérdida Poscaptura y Deterioro (%)</option>
              <option value="histamina">2. Tiempo al Sol e Inocuidad FAO (Horas / Min)</option>
              <option value="costo">3. Costo Mensual FLP vs. 3 Plantas ($ COP)</option>
              <option value="hielo">4. Cobertura Hielo 1:1 en Muelles (%)</option>
              <option value="veda">5. Continuidad en Veda Abril–Junio (kg/día)</option>
              <option value="trazabilidad">6. Hábitats H01/H02 y Trazabilidad DAG (%)</option>
            </select>
            <select id="agy-comp-asoc" class="agy-ctrl-select" onchange="agyRenderizarComparadorAsociacion(this.value)" aria-label="Comparar asociación específica">
              <option value="P05">Inspeccionar Asociación: P05 Puerto Trujillo (Crítico 63 km)</option>
              <option value="P01">Inspeccionar Asociación: P01 ASOPESGA Alto Neblinas</option>
              <option value="P02">Inspeccionar Asociación: P02 Atarraya Verde</option>
              <option value="P03">Inspeccionar Asociación: P03 Asovida San Miguel</option>
              <option value="P04">Inspeccionar Asociación: P04 Asopespor</option>
              <option value="P06">Inspeccionar Asociación: P06 Desembarcadero Central</option>
              <option value="P07">Inspeccionar Asociación: P07 Soplaviento</option>
            </select>
          </div>
        </div>

        <div id="agy-comp-cards-grid" class="agy-bento-grid-3" style="margin-bottom:18px;"></div>
        <div id="agy-comp-asoc-detail" class="agy-plinth-card highlight-iris"></div>
      </div>

      <!-- CONCLUSIÓN FINAL: LO SOLUCIONADO CON CÓDIGO VS LO QUE DEPENDE DE ÁREAS EXTERNAS -->
      <div class="agy-math-showcase" style="margin-bottom:24px;">
        <div class="agy-math-top">
          <div class="agy-math-heading">
            <h3>2. Conclusión de Ingeniería: Problemas Solucionados con Algoritmos vs. Problemas que Dependen de Áreas Externas</h3>
            <p>Por qué ciertas problemáticas de la ECA se resuelven al 100% con nuestro software y por qué otras requieren intervención física, biológica o legal externa.</p>
          </div>
          <span class="agy-badge" style="color:var(--color-saffron-spark);border-color:rgba(255,184,41,0.45);">CRITERIO PROFESIONAL DE ALCANCE</span>
        </div>

        <div class="agy-bento-grid-2" style="margin-bottom:0;">
          <div class="agy-plinth-card highlight-iris">
            <h3 style="color:#10b981;">✓ Problemas 100% Solucionados por Nuestro Software (`G=(V,E)`)</h3>
            <div class="agy-prop-row"><span class="agy-prop-key">1. Ruteo antes del límite FAO (4.0 h)</span><span class="agy-prop-val" style="color:#10b981;">100% resuelto (máx. 2.20 h en P05 con Dijkstra + Poda)</span></div>
            <div class="agy-prop-row"><span class="agy-prop-key">2. Ubicación óptima de planta de frío</span><span class="agy-prop-val" style="color:#10b981;">Solo F02 ahorra $10,000,000 COP/mes (−43.4%)</span></div>
            <div class="agy-prop-row"><span class="agy-prop-key">3. Ruteo y balance de masa Hielo 1:1</span><span class="agy-prop-val" style="color:#10b981;">531.5 kg/día validados al 100% contra capacidad_kg</span></div>
            <div class="agy-prop-row"><span class="agy-prop-key">4. Conmutación en Veda (Abril–Junio)</span><span class="agy-prop-val" style="color:#10b981;">Grafo bitemporal G_veda mantiene 240 kg/día (t ≤ 0.21 h)</span></div>
            <div class="agy-prop-row"><span class="agy-prop-key">5. Desvío de hábitats H01 y H02</span><span class="agy-prop-val" style="color:#10b981;">0 violaciones con evitar_restringidos=True</span></div>
            <div class="agy-prop-row"><span class="agy-prop-key">6. Trazabilidad digital ICA/INVIMA</span><span class="agy-prop-val" style="color:#10b981;">7/7 lotes en DAG de Kahn O(V+E) + Tabla Hash O(1)</span></div>
          </div>

          <div class="agy-plinth-card highlight-saffron">
            <h3 style="color:var(--color-saffron-spark);">⚠ Problemas que No Tienen Solución por Software (Áreas Externas)</h3>
            <p style="font-size:13.5px;margin-bottom:8px;"><strong>¿Por qué no podíamos solucionarlos nosotros como desarrolladores?</strong> Porque son fenómenos bioquímicos, de infraestructura física o de soberanía legal que ocurren fuera del computador:</p>
            <div class="agy-prop-row"><span class="agy-prop-key">1. Nivel de Mercurio (Hg) en peces</span><span class="agy-prop-val">Requiere Espectrometría Química en Laboratorio (GRITOX)</span></div>
            <div class="agy-prop-row"><span class="agy-prop-key">2. Ciclo gonadal y fechas de veda</span><span class="agy-prop-val">Depende de Biología Reproductiva e Hidrología (AUNAP/SUSA)</span></div>
            <div class="agy-prop-row"><span class="agy-prop-key">3. Examen sensorial de ojos/branquias</span><span class="agy-prop-val">Inspección física bromatológica en muelle (INVIMA)</span></div>
            <div class="agy-prop-row"><span class="agy-prop-key">4. Cambiar cajas de madera y techar muelles</span><span class="agy-prop-val">Requiere Obra Civil y Dotación Física (Alcaldía / Cooperativas)</span></div>
            <div class="agy-prop-row"><span class="agy-prop-key">5. Consulta previa en Resguardos Sikuani</span><span class="agy-prop-val">Proceso jurídico-constitucional y social (MinInterior / Kotsala)</span></div>
            <button class="agy-modal-trigger" onclick="agyAbrirModalDetalle('externos')">+ Ver Explicación Detallada de Límites del Software</button>
          </div>
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

        <button class="agy-ctrl-btn" style="color:var(--color-saffron-spark);border-color:rgba(255,184,41,0.45);" onclick="agyAbrirModalConstruccionSubgrafo()">
          📐 Cómo se Construyó este Grafo
        </button>

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

<!-- MODAL EXPANDIBLE (REACT BITS MODAL CARDS STYLE — SCALE WITH MORPHING) -->
<div id="agy-modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="agy-modal-title" onclick="if(event.target===this)agyCerrarModalDetalle()">
  <div class="agy-modal-dialog">
    <button class="agy-modal-close" onclick="agyCerrarModalDetalle()" aria-label="Cerrar ventana de detalle">&times;</button>
    <div id="agy-modal-content"></div>
  </div>
</div>

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
// MOTOR DE CONSTELACIÓN DE PARTÍCULAS TRIANGULARES (ESTÉTICA DALA CON FÍSICA REACTIVA)
// ═══════════════════════════════════════════════════════════════════════
let agyConstelacionAnimId = null;
let agyHeroAnimId = null;

function agyAnimarContadores(container) {{
  if (!container) return;
  const elements = container.querySelectorAll(".agy-stat-num");
  elements.forEach(el => {{
    if (!el.dataset.origHtml) {{
      el.dataset.origHtml = el.innerHTML;
    }}
    const html = el.dataset.origHtml;
    const match = html.match(/^([^0-9]*)([\\d]+(?:[.,]\\d+)?)(.*)$/s);
    if (!match) return;

    const prefix = match[1];
    const numStr = match[2].replace(',', '.');
    const suffix = match[3];
    const targetVal = parseFloat(numStr);
    if (isNaN(targetVal)) return;

    const decimals = numStr.includes('.') ? numStr.split('.')[1].length : 0;
    const duration = 720;
    const startTime = performance.now();

    function step(now) {{
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = (targetVal * ease).toFixed(decimals);
      el.innerHTML = `${{prefix}}${{current}}${{suffix}}`;
      if (progress < 1) {{
        requestAnimationFrame(step);
      }} else {{
        el.innerHTML = html;
      }}
    }}
    requestAnimationFrame(step);
  }});
}}

function agyIniciarConstelacionDala() {{
  const palette = ["#8052ff", "#ffb829", "#15846e", "#2997ff", "#ec4899"];

  // Iniciar Spotlight interactivo en todas las tarjetas
  document.querySelectorAll(".agy-plinth-card, .agy-math-param-card, .agy-stat-card").forEach(card => {{
    card.addEventListener("mousemove", (e) => {{
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      card.style.setProperty("--mouse-x", `${{x}}px`);
      card.style.setProperty("--mouse-y", `${{y}}px`);
    }});
  }});

  // 1. Lienzo de fondo del Explorador de Grafos (#agy-constellation-canvas) con movimiento continuo
  const bgCanvas = document.getElementById("agy-constellation-canvas");
  if (bgCanvas) {{
    const ctx = bgCanvas.getContext("2d");
    const particles = Array.from({{ length: 85 }}, (_, i) => ({{
      x: Math.random(),
      y: Math.random(),
      vx: (Math.random() - 0.5) * 0.0003,
      vy: (Math.random() - 0.5) * 0.0003,
      size: 2.2 + (i % 4) * 1.1,
      angle: Math.random() * Math.PI * 2,
      vRot: (Math.random() - 0.5) * 0.015,
      color: palette[i % palette.length],
      alpha: 0.18 + (i % 5) * 0.06
    }}));

    let mouseBg = {{ x: -1000, y: -1000 }};
    window.addEventListener("mousemove", (e) => {{
      if (!bgCanvas.offsetParent) return;
      const rect = bgCanvas.getBoundingClientRect();
      mouseBg.x = e.clientX - rect.left;
      mouseBg.y = e.clientY - rect.top;
    }});

    function loopBgConstellation() {{
      const w = bgCanvas.clientWidth || 960;
      const h = bgCanvas.clientHeight || 640;
      if (bgCanvas.width !== w || bgCanvas.height !== h) {{
        bgCanvas.width = w;
        bgCanvas.height = h;
      }}
      ctx.clearRect(0, 0, w, h);

      particles.forEach((pt) => {{
        pt.x += pt.vx;
        pt.y += pt.vy;
        pt.angle += pt.vRot;
        if (pt.x < 0) pt.x = 1;
        if (pt.x > 1) pt.x = 0;
        if (pt.y < 0) pt.y = 1;
        if (pt.y > 1) pt.y = 0;

        let px = pt.x * w;
        let py = pt.y * h;

        // Fricción y repulsión suave con el cursor
        const dx = px - mouseBg.x;
        const dy = py - mouseBg.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120 && dist > 1) {{
          const force = (120 - dist) / 120 * 1.8;
          px += (dx / dist) * force * 15;
          py += (dy / dist) * force * 15;
        }}

        ctx.save();
        ctx.translate(px, py);
        ctx.rotate(pt.angle);
        ctx.strokeStyle = pt.color;
        ctx.globalAlpha = pt.alpha;
        ctx.lineWidth = 1.15;
        ctx.beginPath();
        const r = pt.size;
        ctx.moveTo(0, -r);
        ctx.lineTo(r * 0.866, r * 0.5);
        ctx.lineTo(-r * 0.866, r * 0.5);
        ctx.closePath();
        ctx.stroke();
        ctx.restore();
      }});

      if (!agyPrefersReducedMotion()) {{
        agyConstelacionAnimId = requestAnimationFrame(loopBgConstellation);
      }}
    }}
    if (agyConstelacionAnimId) cancelAnimationFrame(agyConstelacionAnimId);
    loopBgConstellation();
  }}

  // 2. Constelación Neural/Topológica en la Portada (#agy-hero-constellation) viva a 60 FPS
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
        vx: 0,
        vy: 0,
        color: AGY_COLORES_NODO[n.tipo] || "#8052ff",
        phase: i * 0.42
      }};
    }});
    const dust = Array.from({{ length: 130 }}, (_, i) => {{
      const a = (i * 2.39996);
      const r = Math.sqrt(i / 130) * 0.44;
      return {{
        x: 0.5 + Math.cos(a) * r * 1.2,
        y: 0.46 + Math.sin(a) * r * 0.88,
        sz: 2.0 + (i % 3) * 1.2,
        rot: i * 0.5,
        vRot: 0.008 + (i % 3) * 0.005,
        col: palette[i % palette.length],
        alpha: 0.28 + (i % 4) * 0.14
      }};
    }});

    let mouseOff = {{ x: 0, y: 0 }};
    let targetMouse = {{ x: 0, y: 0 }};
    heroCanvas.addEventListener("mousemove", (ev) => {{
      const r = heroCanvas.getBoundingClientRect();
      targetMouse.x = ((ev.clientX - r.left) / (r.width || 1) - 0.5) * 32;
      targetMouse.y = ((ev.clientY - r.top) / (r.height || 1) - 0.5) * 32;
    }});
    heroCanvas.addEventListener("mouseleave", () => {{
      targetMouse.x = 0;
      targetMouse.y = 0;
    }});
    heroCanvas.addEventListener("click", () => agyCambiarSlide(4, false));

    function loopHeroConstellation() {{
      const t = performance.now() * 0.001;
      const w = heroCanvas.clientWidth || 480;
      const h = heroCanvas.clientHeight || 390;
      if (heroCanvas.width !== w || heroCanvas.height !== h) {{
        heroCanvas.width = w;
        heroCanvas.height = h;
      }}
      hCtx.clearRect(0, 0, w, h);

      // Interpolación suave del mouse (spring easing)
      mouseOff.x += (targetMouse.x - mouseOff.x) * 0.08;
      mouseOff.y += (targetMouse.y - mouseOff.y) * 0.08;

      // Conexiones tenues palpitantes
      const pulseAlpha = 0.15 + Math.sin(t * 1.5) * 0.06;
      hCtx.strokeStyle = `rgba(128, 82, 255, ${{pulseAlpha}})`;
      hCtx.lineWidth = 0.95;
      for (let i = 0; i < triNodes.length; i++) {{
        const a = triNodes[i];
        const b = triNodes[(i + 3) % triNodes.length];
        const ax = a.bx * w + mouseOff.x * 0.6 + Math.sin(t + a.phase) * 3;
        const ay = a.by * h + mouseOff.y * 0.6 + Math.cos(t + a.phase) * 3;
        const bx = b.bx * w + mouseOff.x * 0.6 + Math.sin(t + b.phase) * 3;
        const by = b.by * h + mouseOff.y * 0.6 + Math.cos(t + b.phase) * 3;
        hCtx.beginPath();
        hCtx.moveTo(ax, ay);
        hCtx.lineTo(bx, by);
        hCtx.stroke();
      }}

      // Nube de triángulos Dala giratorios
      dust.forEach((d, idx) => {{
        d.rot += d.vRot;
        const px = d.x * w + mouseOff.x * ((idx % 3) * 0.4) + Math.cos(t * 0.8 + idx) * 2;
        const py = d.y * h + mouseOff.y * ((idx % 3) * 0.4) + Math.sin(t * 0.8 + idx) * 2;
        hCtx.save();
        hCtx.translate(px, py);
        hCtx.rotate(d.rot);
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

      // Vértices principales con halo luminoso
      triNodes.forEach((nd) => {{
        const px = nd.bx * w + mouseOff.x + Math.sin(t * 1.2 + nd.phase) * 4;
        const py = nd.by * h + mouseOff.y + Math.cos(t * 1.2 + nd.phase) * 4;
        hCtx.save();
        hCtx.translate(px, py);
        
        // Halo de resplandor
        hCtx.beginPath();
        hCtx.arc(0, 0, 11, 0, Math.PI * 2);
        hCtx.fillStyle = nd.color;
        hCtx.globalAlpha = 0.18 + Math.sin(t * 2 + nd.phase) * 0.1;
        hCtx.fill();

        // Núcleo
        hCtx.globalAlpha = 1;
        hCtx.beginPath();
        hCtx.arc(0, 0, 5.5, 0, Math.PI * 2);
        hCtx.fillStyle = "#000000";
        hCtx.strokeStyle = nd.color;
        hCtx.lineWidth = 2.2;
        hCtx.fill();
        hCtx.stroke();

        hCtx.fillStyle = "#f5f5f7";
        hCtx.font = "600 10px 'JetBrains Mono', monospace";
        hCtx.fillText(nd.id, 9, 3.5);
        hCtx.restore();
      }});

      if (!agyPrefersReducedMotion()) {{
        agyHeroAnimId = requestAnimationFrame(loopHeroConstellation);
      }}
    }}
    if (agyHeroAnimId) cancelAnimationFrame(agyHeroAnimId);
    loopHeroConstellation();
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

  // 5. Gráficos de FLP, Workspace Detallado FLP y Tiempos Térmicos (Diapositiva 7)
  agyInicializarHoverTooltipsFLP();
  agyRenderizarWorkspaceFLP();
  agyRenderizarGraficosSlide7("PG");

  // 6. Simulador Térmico Crosshair (Slide 1) y Comparador Antes vs Después (Slide 8)
  agyActualizarCrosshairTermico(2.20);
  agyRenderizarComparadorInteractivo();
  agyRenderizarComparadorAsociacion("P05");
}}

function agyCambiarPasoConstruccion(idx) {{
  for (let i = 0; i < 4; i++) {{
    document.getElementById(`agy-step-btn-${{i}}`)?.classList.toggle("active", i === idx);
    document.getElementById(`agy-step-pane-${{i}}`)?.classList.toggle("active", i === idx);
  }}
}}

/* ═══════════════════════════════════════════════════════════════
   SIMULADOR TÉRMICO v2 — Canvas Highcharts-style con Plot Bands,
   Tooltip flotante, Spline suavizado y Leyenda interactiva
   ═══════════════════════════════════════════════════════════════ */
const agyThermState = {{
  t: 2.20,
  series: {{ sinFrio: true, zonaD: true, conF02: true }},
  hoverT: null,
  hoverSerie: null,
  animProgress: 0,
  animFrame: null
}};

const AGY_THERM_SERIES_META = {{
  sinFrio: {{
    key: "sinFrio",
    nombre: "Sin Solución (30 °C)",
    tasa: "2.5% / h · Exposición ambiente",
    fn: agyCalcSinFrio,
    color: "#ff6b2b",
    halo: "rgba(255,107,43,0.24)",
    r: 6
  }},
  zonaD: {{
    key: "zonaD",
    nombre: "Zona Crítica D (×1.5)",
    tasa: "3.75% / h · Tramo de deterioro acelerado",
    fn: agyCalcZonaD,
    color: "#dc5000",
    halo: "rgba(220,80,0,0.24)",
    r: 5.5
  }},
  conF02: {{
    key: "conF02",
    nombre: "Con Centro F02 (0 °C)",
    tasa: "0.25% / h · Cadena de frío optimizada",
    fn: agyCalcConF02,
    color: "#10b981",
    halo: "rgba(16,185,129,0.24)",
    r: 6
  }}
}};

function agySetPreset(val, btn) {{
  const slider = document.getElementById("agy-sim-time-slider");
  if (slider) slider.value = val;
  document.querySelectorAll(".agy-preset-btn").forEach(b => b.classList.remove("active-preset"));
  if (btn) btn.classList.add("active-preset");
  agyActualizarCrosshairTermico(val);
}}

function agyToggleSerie(key, el) {{
  agyThermState.series[key] = !agyThermState.series[key];
  el.classList.toggle("disabled", !agyThermState.series[key]);
  agyDibujarChartTermico();
}}

// Modelos de degradación (misma lógica existente)
function agyCalcSinFrio(h) {{ return Math.min(100, h * 2.5); }}
function agyCalcZonaD(h) {{ return Math.min(100, h * 3.75); }}
function agyCalcConF02(h) {{
  const tAntes = Math.min(h, 0.85);
  const tDespues = Math.max(0, h - tAntes);
  return Math.min(100, tAntes * 2.5 + tDespues * 0.25);
}}

function agyDibujarChartTermico() {{
  const canvas = document.getElementById("agy-therm-canvas");
  if (!canvas) return;
  const wrap = document.getElementById("agy-therm-chart-wrap");
  const rect = wrap.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const W = rect.width;
  const H = 320;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  // Layout del área de dibujo
  const pad = {{ l: 56, r: 28, t: 24, b: 42 }};
  const pW = W - pad.l - pad.r;
  const pH = H - pad.t - pad.b;
  const maxPct = 45;
  const maxH = 10;

  const xOf = (h) => pad.l + (h / maxH) * pW;
  const yOf = (pct) => pad.t + pH - (Math.min(maxPct, pct) / maxPct) * pH;

  // Fondo
  ctx.fillStyle = "#08080b";
  ctx.fillRect(0, 0, W, H);

  // ── PLOT BANDS (franjas horizontales tipo Highcharts) ──
  const bands = [
    {{ from: 0, to: 5, color: "rgba(16,185,129,0.04)", label: "Óptimo (< 5%)" }},
    {{ from: 5, to: 10, color: "rgba(255,184,41,0.04)", label: "Leve (5–10%)" }},
    {{ from: 10, to: 20, color: "rgba(255,107,43,0.04)", label: "Moderada (10–20%)" }},
    {{ from: 20, to: 30, color: "rgba(220,80,0,0.06)", label: "Severa (20–30%)" }},
    {{ from: 30, to: 45, color: "rgba(220,30,0,0.06)", label: "Pérdida total (> 30%)" }}
  ];
  bands.forEach((b, i) => {{
    const y1 = yOf(b.to);
    const y2 = yOf(b.from);
    ctx.fillStyle = b.color;
    ctx.fillRect(pad.l, y1, pW, y2 - y1);
    // Etiqueta de banda
    ctx.fillStyle = "rgba(255,255,255,0.18)";
    ctx.font = "500 10px 'Plus Jakarta Sans', sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(b.label, W - pad.r - 6, y1 + 14);
  }});

  // ── GRID LINES (horizontales sutiles) ──
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth = 1;
  for (let p = 0; p <= maxPct; p += 5) {{
    const y = yOf(p);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(W - pad.r, y);
    ctx.stroke();
  }}

  // ── GRID LINES (verticales cada hora) ──
  for (let h = 0; h <= maxH; h++) {{
    const x = xOf(h);
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + pH);
    ctx.strokeStyle = h === 4 ? "rgba(255,184,41,0.25)" : "rgba(255,255,255,0.04)";
    ctx.lineWidth = h === 4 ? 1.5 : 1;
    ctx.stroke();
  }}

  // ── ZONA ROJA FAO (> 4.0 h) ──
  const xFAO = xOf(4.0);
  ctx.fillStyle = "rgba(220,80,0,0.06)";
  ctx.fillRect(xFAO, pad.t, xOf(10) - xFAO, pH);
  // Etiqueta FAO
  ctx.save();
  ctx.fillStyle = "#ffb829";
  ctx.font = "700 10px 'JetBrains Mono', monospace";
  ctx.textAlign = "left";
  ctx.fillText("LÍMITE FAO 4.0 h → RIESGO HISTAMINA", xFAO + 6, pad.t + 14);
  ctx.restore();
  // Línea punteada vertical FAO
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = "rgba(255,184,41,0.5)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(xFAO, pad.t);
  ctx.lineTo(xFAO, pad.t + pH);
  ctx.stroke();
  ctx.setLineDash([]);

  // ── GENERAR PUNTOS DE LAS TRES CURVAS ──
  const pts = {{ sinFrio: [], zonaD: [], conF02: [] }};
  for (let h = 0; h <= maxH; h += 0.1) {{
    const hR = Math.round(h * 10) / 10;
    pts.sinFrio.push({{ x: xOf(hR), y: yOf(agyCalcSinFrio(hR)), h: hR, v: agyCalcSinFrio(hR) }});
    pts.zonaD.push({{ x: xOf(hR), y: yOf(agyCalcZonaD(hR)), h: hR, v: agyCalcZonaD(hR) }});
    pts.conF02.push({{ x: xOf(hR), y: yOf(agyCalcConF02(hR)), h: hR, v: agyCalcConF02(hR) }});
  }}

  // ── FUNCIÓN: DIBUJAR CURVA SPLINE CON ÁREA ──
  function drawSplineArea(points, strokeColor, fillColor, lineWidth, dashed) {{
    if (points.length < 2) return;
    // Área rellena
    ctx.beginPath();
    ctx.moveTo(points[0].x, pad.t + pH);
    ctx.lineTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {{
      const prev = points[i - 1];
      const curr = points[i];
      const cpx = (prev.x + curr.x) / 2;
      ctx.bezierCurveTo(cpx, prev.y, cpx, curr.y, curr.x, curr.y);
    }}
    ctx.lineTo(points[points.length - 1].x, pad.t + pH);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();

    // Línea de la curva
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {{
      const prev = points[i - 1];
      const curr = points[i];
      const cpx = (prev.x + curr.x) / 2;
      ctx.bezierCurveTo(cpx, prev.y, cpx, curr.y, curr.x, curr.y);
    }}
    if (dashed) ctx.setLineDash([6, 4]);
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
    if (dashed) ctx.setLineDash([]);
  }}

  // ── DIBUJAR LAS SERIES (si están activas) ──
  const hs = agyThermState.hoverSerie;
  if (agyThermState.series.sinFrio) {{
    drawSplineArea(pts.sinFrio, "#ff6b2b", "rgba(255,107,43,0.12)", hs === "sinFrio" ? 3.8 : 2.6, false);
  }}
  if (agyThermState.series.zonaD) {{
    drawSplineArea(pts.zonaD, "#dc5000", "rgba(220,80,0,0.06)", hs === "zonaD" ? 3.0 : 1.8, true);
  }}
  if (agyThermState.series.conF02) {{
    drawSplineArea(pts.conF02, "#10b981", "rgba(16,185,129,0.10)", hs === "conF02" ? 3.8 : 2.6, false);
  }}

  // ── CROSSHAIR E INDICADORES (BOLITAS QUE SIGUEN AL MOUSE O AL SLIDER) ──
  const activeT = agyThermState.hoverT !== null ? agyThermState.hoverT : agyThermState.t;
  const cx = xOf(activeT);

  // Línea guía vertical
  ctx.strokeStyle = hs ? AGY_THERM_SERIES_META[hs].color + "99" : "rgba(128,82,255,0.65)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(cx, pad.t);
  ctx.lineTo(cx, pad.t + pH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Puntos de intersección (las bolitas se desplazan en vivo con el mouse)
  Object.values(AGY_THERM_SERIES_META).forEach(s => {{
    if (!agyThermState.series[s.key]) return;
    const isHovered = hs === s.key;
    const isDimmed = hs !== null && !isHovered;
    const pct = s.fn(activeT);
    const cy = yOf(pct);
    const rad = isHovered ? s.r + 2.5 : (isDimmed ? s.r - 1.5 : s.r);

    ctx.save();
    if (isDimmed) ctx.globalAlpha = 0.45;

    // Halo exterior (más grande y brillante en la línea bajo el cursor)
    ctx.beginPath();
    ctx.arc(cx, cy, isHovered ? rad + 7 : rad + 4, 0, Math.PI * 2);
    ctx.fillStyle = s.halo;
    ctx.fill();

    // Línea horizontal guía hacia el eje Y cuando el mouse está sobre esta curva
    if (isHovered) {{
      ctx.setLineDash([2, 3]);
      ctx.strokeStyle = s.color + "88";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.l, cy);
      ctx.lineTo(cx, cy);
      ctx.stroke();
      ctx.setLineDash([]);
    }}

    // Bolita principal
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, Math.PI * 2);
    ctx.fillStyle = s.color;
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = isHovered ? 2.5 : 1.8;
    ctx.stroke();
    ctx.restore();
  }});

  // ── EJES ──
  ctx.fillStyle = "#6e6e73";
  ctx.font = "500 10px 'JetBrains Mono', monospace";
  ctx.textAlign = "right";
  for (let p = 0; p <= maxPct; p += 5) {{
    ctx.fillText(p + "%", pad.l - 8, yOf(p) + 3.5);
  }}
  ctx.textAlign = "center";
  for (let h = 0; h <= maxH; h++) {{
    ctx.fillStyle = h === 4 ? "#ffb829" : "#6e6e73";
    ctx.fillText(h + ".0 h", xOf(h), H - pad.b + 18);
  }}

  // Etiqueta eje Y
  ctx.save();
  ctx.translate(14, pad.t + pH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#6e6e73";
  ctx.font = "600 10px 'Plus Jakarta Sans', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Degradación (%)", 0, 0);
  ctx.restore();

  // Etiqueta eje X
  ctx.fillStyle = "#6e6e73";
  ctx.font = "600 10px 'Plus Jakarta Sans', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Tiempo de exposición sin refrigeración (horas)", pad.l + pW / 2, H - 4);

  // Hitos clave en eje X (anotaciones)
  const hitos = [
    {{ h: 0.85, label: "F02 abate", color: "#10b981" }},
    {{ h: 2.20, label: "P05 máx", color: "#2997ff" }},
    {{ h: 9.48, label: "Bogotá", color: "#ff6b2b" }}
  ];
  hitos.forEach(m => {{
    const mx = xOf(m.h);
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = m.color + "55";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mx, pad.t + pH - 4);
    ctx.lineTo(mx, pad.t + pH + 8);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = m.color;
    ctx.font = "700 9px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(m.label, mx, pad.t + pH + 32);
  }});
}}

// ── HOVER / TOOLTIP flotante sobre el Canvas (solo al pasar sobre la línea + seguimiento continuo) ──
(function() {{
  let canvasReady = false;
  function setupCanvasHover() {{
    const canvas = document.getElementById("agy-therm-canvas");
    const tooltip = document.getElementById("agy-therm-tooltip");
    const wrap = document.getElementById("agy-therm-chart-wrap");
    if (!canvas || !tooltip || !wrap) {{
      if (!canvasReady) setTimeout(setupCanvasHover, 300);
      return;
    }}
    canvasReady = true;
    const pad = {{ l: 56, r: 28, t: 24, b: 42 }};
    const maxPct = 45;
    const HIT_TOLERANCE_PX = 24;

    canvas.addEventListener("mousemove", function(e) {{
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const pW = rect.width - pad.l - pad.r;
      const pH = rect.height - pad.t - pad.b;

      if (mx < pad.l || mx > rect.width - pad.r || my < pad.t || my > pad.t + pH) {{
        tooltip.classList.remove("visible");
        agyThermState.hoverT = null;
        agyThermState.hoverSerie = null;
        canvas.style.cursor = "crosshair";
        agyDibujarChartTermico();
        return;
      }}

      // Hora continua según la posición X del ratón para que las bolitas se deslicen fluidamente
      const hVal = Math.max(0, Math.min(10, ((mx - pad.l) / pW) * 10));
      const yOf = (pct) => pad.t + pH - (Math.min(maxPct, pct) / maxPct) * pH;

      // Detectar cuál curva está directamente bajo el cursor (distancia vertical my -> cy)
      let closestKey = null;
      let minDist = Infinity;
      Object.values(AGY_THERM_SERIES_META).forEach(s => {{
        if (!agyThermState.series[s.key]) return;
        const cy = yOf(s.fn(hVal));
        const dist = Math.abs(my - cy);
        if (dist < minDist) {{
          minDist = dist;
          closestKey = s.key;
        }}
      }});

      const isOverLine = closestKey !== null && minDist <= HIT_TOLERANCE_PX;

      // Actualizar estado para que los indicadores (bolitas), el slider y las tarjetas sigan al mouse en vivo
      agyThermState.hoverT = hVal;
      agyThermState.hoverSerie = isOverLine ? closestKey : null;
      agyThermState.t = hVal;

      const slider = document.getElementById("agy-sim-time-slider");
      if (slider) slider.value = hVal.toFixed(2);
      const lbl = document.getElementById("agy-sim-time-lbl");
      if (lbl) lbl.textContent = `${{hVal.toFixed(2)}} h (${{Math.round(hVal * 60)}} min)`;

      agyRenderizarTarjetasTermicas(hVal);
      agyDibujarChartTermico();

      // Mostrar el cuadro de información SOLO cuando el cursor pasa por encima de la línea
      if (!isOverLine) {{
        tooltip.classList.remove("visible");
        canvas.style.cursor = "crosshair";
        return;
      }}

      canvas.style.cursor = "pointer";
      const sMeta = AGY_THERM_SERIES_META[closestKey];
      const pctVal = sMeta.fn(hVal);
      const prodTotal = 265.75;
      const kgPerdidos = prodTotal * (pctVal / 100);
      const copPerdidos = Math.round(kgPerdidos * 12000);

      const faoStatus = hVal > 4.0
        ? `<div style="margin-top:6px;padding:3px 8px;background:rgba(220,80,0,0.16);border-radius:6px;color:#ff6b2b;font-size:10px;font-weight:700;">⚠ SUPERA LÍMITE FAO (240 min)</div>`
        : `<div style="margin-top:6px;padding:3px 8px;background:rgba(16,185,129,0.14);border-radius:6px;color:#10b981;font-size:10px;font-weight:700;">✓ INOCUO (DENTRO DE LÍMITE FAO)</div>`;

      tooltip.innerHTML = `
        <div class="tt-title" style="display:flex;align-items:center;gap:7px;">
          <span class="tt-dot" style="background:${{sMeta.color}}"></span>
          <span>${{sMeta.nombre}}</span>
        </div>
        <div style="font-size:10.5px;color:var(--color-ash);margin-bottom:6px;">${{sMeta.tasa}}</div>
        <div class="tt-row"><span>Tiempo (t):</span><span class="tt-val" style="color:var(--color-bone-white)">${{hVal.toFixed(2)}} h (${{Math.round(hVal * 60)}} min)</span></div>
        <div class="tt-row"><span>Degradación:</span><span class="tt-val" style="color:${{sMeta.color}}">${{pctVal.toFixed(2)}}%</span></div>
        <div class="tt-row"><span>Merma física:</span><span class="tt-val">${{kgPerdidos.toFixed(1)}} kg / 265.8 kg</span></div>
        <div class="tt-row"><span>Impacto económico:</span><span class="tt-val">$${{copPerdidos.toLocaleString()}} COP</span></div>
        ${{faoStatus}}
      `;
      tooltip.classList.add("visible");

      // Posicionar el tooltip anclado a la bolita activa sobre la línea
      const cx = pad.l + (hVal / 10) * pW;
      const cy = yOf(pctVal);
      let tx = cx + 16;
      let ty = cy - 42;
      if (tx + 240 > rect.width) tx = cx - 252;
      if (ty + 155 > rect.height) ty = rect.height - 162;
      if (ty < 8) ty = 8;
      tooltip.style.left = tx + "px";
      tooltip.style.top = ty + "px";
    }});

    canvas.addEventListener("mouseleave", function() {{
      tooltip.classList.remove("visible");
      agyThermState.hoverT = null;
      agyThermState.hoverSerie = null;
      canvas.style.cursor = "crosshair";
      agyDibujarChartTermico();
    }});

    // Click en el canvas fija el tiempo seleccionado
    canvas.addEventListener("click", function(e) {{
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const pW = rect.width - pad.l - pad.r;
      if (mx >= pad.l && mx <= rect.width - pad.r) {{
        const hVal = Math.max(0, Math.min(10, ((mx - pad.l) / pW) * 10));
        const slider = document.getElementById("agy-sim-time-slider");
        if (slider) slider.value = hVal.toFixed(2);
        document.querySelectorAll(".agy-preset-btn").forEach(b => b.classList.remove("active-preset"));
        agyActualizarCrosshairTermico(hVal);
      }}
    }});
  }}
  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", setupCanvasHover);
  }} else {{
    setTimeout(setupCanvasHover, 100);
  }}
}})();

function agyRenderizarTarjetasTermicas(t) {{
  const prodTotal = 265.75;
  const pctSinFrio = Math.min(100, t * 2.5);
  const pctZonaD = Math.min(100, t * 3.75);
  const tAntesFrio = Math.min(t, 0.85);
  const tEnFrio = Math.max(0, t - tAntesFrio);
  const pctConF02 = Math.min(100, tAntesFrio * 2.5 + tEnFrio * 0.25);

  const kgSin = prodTotal * (pctSinFrio / 100);
  const kgCon = prodTotal * (pctConF02 / 100);
  const copPerdidos = kgSin * 12000;
  const copAhorrados = (kgSin - kgCon) * 12000;
  const violaFAO = t > 4.0;

  const tiLive = document.getElementById("agy-fcard-ti-live");
  if (tiLive) tiLive.textContent = `${{t.toFixed(2)}} h (${{Math.round(t * 60)}} min)`;

  const cardsWrap = document.getElementById("agy-sim-readout-cards");
  if (cardsWrap) {{
    cardsWrap.innerHTML = `
      <div class="agy-readout-card">
        <div class="agy-readout-val" style="color:#ff6b2b;">${{pctSinFrio.toFixed(1)}}%</div>
        <div class="agy-readout-label">Merma sin solución (30 °C)</div>
        <div class="agy-readout-sub">${{kgSin.toFixed(1)}} kg perdidos · $${{Math.round(copPerdidos).toLocaleString()}} COP/día.
En Zona D: ${{pctZonaD.toFixed(1)}}%.</div>
      </div>
      <div class="agy-readout-card">
        <div class="agy-readout-val" style="color:#10b981;">${{pctConF02.toFixed(1)}}%</div>
        <div class="agy-readout-label">Merma con F02 (0 °C)</div>
        <div class="agy-readout-sub">${{kgCon.toFixed(1)}} kg perdidos · Vida útil extendida a 360 h.</div>
      </div>
      <div class="agy-readout-card">
        <div class="agy-readout-val" style="color:var(--color-saffron-spark);">$${{Math.round(copAhorrados).toLocaleString()}}</div>
        <div class="agy-readout-label">COP ahorrados / día</div>
        <div class="agy-readout-sub">${{(kgSin - kgCon).toFixed(1)}} kg/día rescatados de descomposición.</div>
      </div>
      <div class="agy-readout-card" style="border-color:${{violaFAO ? 'rgba(220,80,0,0.3)' : 'rgba(16,185,129,0.2)'}};">
        <div class="agy-readout-val" style="font-size:20px;color:${{violaFAO ? '#ff6b2b' : '#10b981'}};">${{violaFAO ? '⚠ HISTAMINA CRÍTICA' : '✓ INOCUO (FAO OK)'}}</div>
        <div class="agy-readout-label">Estado sanitario a t = ${{t.toFixed(2)}} h</div>
        <div class="agy-readout-sub">${{violaFAO ? 'Supera 240 min sin frío: no apto para consumo.' : 'Dentro de la ventana FAO de 240 min.'}}</div>
      </div>
    `;
  }}
}}

function agyToggleFormulaTermica(forzarAbrir) {{
  const drawer = document.getElementById("agy-therm-formula-drawer");
  const btn = document.getElementById("agy-therm-formula-btn");
  const hint = document.getElementById("agy-therm-formula-hint-txt");
  if (!drawer || !btn) return;
  const open = typeof forzarAbrir === "boolean" ? forzarAbrir : !drawer.classList.contains("open");
  drawer.classList.toggle("open", open);
  btn.classList.toggle("is-open", open);
  btn.setAttribute("aria-expanded", open ? "true" : "false");
  if (hint) {{
    hint.textContent = open
      ? "Ocultar desglose matemático de la ecuación (Haz clic aquí)"
      : "Haz clic en la fórmula o en sus términos para desglosar variables";
  }}
}}

function agySeleccionarTerminoFormula(termKey) {{
  document.querySelectorAll("#agy-therm-formula-drawer .agy-term-pill").forEach(p => {{
    p.classList.toggle("active", p.getAttribute("data-term") === termKey);
  }});
  document.querySelectorAll("#agy-formula-cards-grid .agy-formula-card").forEach(c => {{
    const key = c.getAttribute("data-fcard");
    if (termKey === "all") {{
      c.style.display = "";
      c.classList.remove("highlighted");
    }} else {{
      c.style.display = "";
      c.classList.toggle("highlighted", key === termKey);
    }}
  }});
}}

function agyEnfocarVariableFormula(termKey) {{
  agyToggleFormulaTermica(true);
  agySeleccionarTerminoFormula(termKey);
}}

function agyActualizarCrosshairTermico(horasVal) {{
  const t = Math.max(0, Math.min(10, parseFloat(horasVal) || 0));
  agyThermState.t = t;
  const lbl = document.getElementById("agy-sim-time-lbl");
  if (lbl) lbl.textContent = `${{t.toFixed(2)}} h (${{Math.round(t * 60)}} min)`;

  agyDibujarChartTermico();
  agyRenderizarTarjetasTermicas(t);
}}

// Redimensionar el canvas al cambiar tamaño de ventana
window.addEventListener("resize", function() {{
  if (document.getElementById("agy-therm-canvas")) agyDibujarChartTermico();
}});


function agyRenderizarComparadorInteractivo() {{
  const filtro = document.getElementById("agy-comp-filter")?.value || "all";
  const grid = document.getElementById("agy-comp-cards-grid");
  if (!grid) return;

  const items = [
    {{
      id: "merma",
      titulo: "1. Pérdida Poscaptura en Ruta Nacional",
      antesTxt: "36.1% sim. / hasta 40.0% campo (95.9 kg/día perdidos)",
      despuesTxt: "1.3% local (6.3 kg) / 17.1% a Bogotá con F02",
      pctAntes: 90,
      pctDespues: 18,
      delta: "−52.6% a −94.2% merma",
      explicacion: "Al enfriar a 0 °C en F02 antes de salir por la Ruta 40, se rescatan más de 50 kg/día de pescado fresco."
    }},
    {{
      id: "histamina",
      titulo: "2. Exposición al Sol y Riesgo de Histamina (FAO ≤ 4.0 h)",
      antesTxt: "9.48 h (569 min sin frío → 0% cumplimiento FAO)",
      despuesTxt: "Máx. 2.20 h (132 min en P05 → 100% cumplimiento 7/7)",
      pctAntes: 100,
      pctDespues: 23,
      delta: "−76.8% tiempo sin frío",
      explicacion: "Dijkstra con poda sanitaria garantiza que las 7 asociaciones toquen frío con +108 a +235 min de margen."
    }},
    {{
      id: "costo",
      titulo: "3. Costo Mensual de Infraestructura Frigorífica (FLP)",
      antesTxt: "$23,023,272 COP/mes (Abriendo 3 plantas F01+F02+F03)",
      despuesTxt: "$13,023,272 COP/mes (Configuración óptima Solo F02)",
      pctAntes: 100,
      pctDespues: 56,
      delta: "Ahorra $10.0M COP/mes (−43.4%)",
      explicacion: "El optimizador FLP demuestra que F02 cubre el 100% de la cuenca usando el 5.3% de su capacidad (5,000 kg)."
    }},
    {{
      id: "hielo",
      titulo: "4. Suministro de Hielo 1:1 en Desembarcaderos",
      antesTxt: "0:1 (Sin hielo en canoas; masa sin refrigerar 265.8 kg)",
      despuesTxt: "1:1 FAO (265.8 kg hielo/día + despacho inverso desde F02)",
      pctAntes: 8,
      pctDespues: 100,
      delta: "100% factibilidad capacidad_kg",
      explicacion: "La masa bruta duplicada (531.5 kg/día) respeta el 100% de la capacidad de lanchas y camiones."
    }},
    {{
      id: "veda",
      titulo: "5. Producción Formal Durante Veda (Abril–Junio)",
      antesTxt: "0.0 kg/día fluviales (100% parálisis extractiva por 3 meses)",
      despuesTxt: "240.0 kg/día en Grafo Bitemporal G_veda (P_PISC1..3)",
      pctAntes: 5,
      pctDespues: 90,
      delta: "90.3% continuidad productiva",
      explicacion: "Conmuta orígenes a 3 granjas de piscicultura nativa con acceso terrestre a F02 en t_máx = 0.21 h."
    }},
    {{
      id: "trazabilidad",
      titulo: "6. Protección de Hábitats (H01/H02) y Trazabilidad ICA/INVIMA",
      antesTxt: "Rutas sin control sobre humedales y 0% lotes auditados",
      despuesTxt: "0 violaciones en H01/H02 + 7/7 lotes en DAG Kahn + Hash O(1)",
      pctAntes: 10,
      pctDespues: 100,
      delta: "100% trazabilidad y protección H",
      explicacion: "Cada lote (ECA-2026-P01..P07) registra su integral térmica °C·h en tiempo constante O(1)."
    }}
  ];

  const visibles = filtro === "all" ? items : items.filter(x => x.id === filtro);
  grid.innerHTML = visibles.map(it => `
    <div class="agy-comp-card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
        <div class="agy-stat-title">${{it.titulo}}</div>
        <span class="agy-delta-pill">${{it.delta}}</span>
      </div>
      <div class="agy-comp-dual-bar">
        <div style="font-size:12px;color:var(--color-ember-accent);font-family:var(--font-mono);">ANTES (SIN SOLUCIÓN): ${{it.antesTxt}}</div>
        <div class="agy-comp-track"><div class="agy-comp-fill-before" style="width:${{it.pctAntes}}%"></div></div>
        <div style="font-size:12px;color:#10b981;font-family:var(--font-mono);margin-top:4px;">CON SOLUCIÓN G=(V,E): ${{it.despuesTxt}}</div>
        <div class="agy-comp-track"><div class="agy-comp-fill-after" style="width:${{it.pctDespues}}%"></div></div>
      </div>
      <p style="font-size:13px;">${{it.explicacion}}</p>
    </div>
  `).join("");
}}

function agyRenderizarComparadorAsociacion(asocId) {{
  const box = document.getElementById("agy-comp-asoc-detail");
  if (!box) return;
  const itBog = (AGY_DATA.itinerarios || []).find(x => x.origen_id === asocId) || (AGY_DATA.itinerarios || [])[0];
  const ecaInoc = ((AGY_DATA.eca || {{}}).inocuidad_histamina || {{}}).asociaciones || [];
  const inocRow = ecaInoc.find(x => x.origen_id === asocId) || {{ tiempo_sin_frio_h: 0.48, tiempo_sin_frio_min: 28.8, margen_seguridad_min: 211.2, riesgo_histamina: "CONTROLADO" }};
  if (!itBog) return;

  const tSinSolBog = (itBog.tiempo_total_h || 6.78);
  const mermaSinSolPct = Math.min(40.0, tSinSolBog * 3.2);
  const kgPerdSinSol = (itBog.kg_inicial * mermaSinSolPct / 100);

  box.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
      <div>
        <span class="agy-badge" style="color:var(--color-saffron-spark);margin-bottom:6px;">AUDITORÍA INDIVIDUAL ANTES VS. DESPUÉS · ${{itBog.origen_id}}</span>
        <h3>${{itBog.origen_nombre}} (${{itBog.kg_inicial}} kg/día + ${{itBog.kg_inicial}} kg Hielo 1:1 = ${{(itBog.kg_inicial * 2).toFixed(1)}} kg brutos)</h3>
      </div>
      <button class="agy-pill-ghost" onclick="agySaltarAItinerario('${{itBog.origen_id}}')">Ver Ruta de ${{itBog.origen_id}} en el Grafo →</button>
    </div>
    <div class="agy-bento-grid-4" style="margin-top:12px;margin-bottom:0;">
      <div class="agy-plinth-card obsidian">
        <div class="agy-stat-title" style="color:var(--color-ember-accent);">Tiempo al Sol (Antes vs. Después)</div>
        <div style="font-family:var(--font-mono);font-size:14px;margin-top:4px;">Antes: <strong style="color:var(--color-ember-accent);">${{tSinSolBog.toFixed(2)}} h (${{Math.round(tSinSolBog*60)}} min)</strong><br>Con F02: <strong style="color:#10b981;">${{inocRow.tiempo_sin_frio_h.toFixed(2)}} h (${{inocRow.tiempo_sin_frio_min}} min)</strong></div>
      </div>
      <div class="agy-plinth-card obsidian">
        <div class="agy-stat-title" style="color:#10b981;">Margen contra Histamina (FAO 240 min)</div>
        <div style="font-family:var(--font-mono);font-size:14px;margin-top:4px;">Antes: <strong style="color:var(--color-ember-accent);">VIOLA (+${{Math.round((tSinSolBog-4)*60)}} min)</strong><br>Con F02: <strong style="color:#10b981;">+${{inocRow.margen_seguridad_min}} min a favor (${{inocRow.riesgo_histamina}})</strong></div>
      </div>
      <div class="agy-plinth-card obsidian">
        <div class="agy-stat-title" style="color:var(--color-saffron-spark);">Merma de Pescado en Ruta</div>
        <div style="font-family:var(--font-mono);font-size:14px;margin-top:4px;">Sin frío: <strong style="color:var(--color-ember-accent);">${{mermaSinSolPct.toFixed(1)}}% (${{kgPerdSinSol.toFixed(1)}} kg)</strong><br>Con F02 local: <strong style="color:#10b981;">${{(inocRow.tiempo_sin_frio_h * 2.5).toFixed(1)}}% (${{(itBog.kg_inicial * inocRow.tiempo_sin_frio_h * 0.025).toFixed(1)}} kg)</strong></div>
      </div>
      <div class="agy-plinth-card obsidian">
        <div class="agy-stat-title" style="color:var(--color-electric-iris);">Ruta Multimodal Certificada (DAG)</div>
        <div style="font-family:var(--font-mono);font-size:12.5px;margin-top:4px;color:var(--color-warm-cream);">${{(itBog.paradas || []).join(" → ")}}</div>
      </div>
    </div>
  `;
}}

const AGY_MODALES_INFO = {{
  kpi_prod: {{
    tag: "INDICADOR MAESTRO 1 · PRODUCCIÓN DIARIA EXPUESTA (265.75 kg/día)",
    titulo: "Desglose Oficial de Producción por Asociación Pesquera y Justificación AUNAP / ECA",
    cuerpo: `
      <p>La carga total inyectada diariamente al grafo logístico <span class="agy-tex"><var>G</var> <span class="agy-op">=</span> (<var>V</var>, <var>E</var>)</span> proviene de los <strong>7 puntos de pesca artesanal (<span class="agy-tex"><var>P</var><sub>01</sub> &hellip; <var>P</var><sub>07</sub></span>)</strong> ubicados sobre las cuencas de los ríos Meta, Manacacías e Iteviare:</p>
      <div style="overflow-x:auto;margin-top:14px;">
        <table class="agy-table" style="width:100%;border-collapse:collapse;">
          <thead>
            <tr>
              <th style="text-align:left;padding:10px 12px;">Nodo</th>
              <th style="text-align:left;padding:10px 12px;">Asociación / Punto</th>
              <th style="text-align:right;padding:10px 12px;">Producción (kg/día)</th>
              <th style="text-align:left;padding:10px 12px;">Origen / Justificación del dato</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:10px 12px;"><strong style="color:var(--col-pesca);">P01</strong></td>
              <td style="padding:10px 12px;"><strong>ASOPESGA (Alto Neblinas)</strong></td>
              <td style="padding:10px 12px;text-align:right;font-family:var(--font-mono);font-weight:700;color:var(--color-saffron-spark);">65.75</td>
              <td style="padding:10px 12px;">Dato ancla histórico de la AUNAP: Corresponde al registro histórico formal de ASOPESGA de <strong>24 toneladas/año</strong> de captura extractiva convertido a tasa diaria:
                <div class="agy-tex-block" style="margin:6px 0 2px;padding:8px 14px;font-size:16px;">
                  <span class="agy-tex">
                    <span class="agy-frac"><span class="agy-frac-num">24,000 kg</span><span class="agy-frac-den">365 días</span></span>
                    <span class="agy-op">&approx;</span> 65.7534 kg/día
                  </span>
                </div>
              </td>
            </tr>
            <tr>
              <td style="padding:10px 12px;"><strong style="color:var(--col-pesca);">P02</strong></td>
              <td style="padding:10px 12px;"><strong>Atarraya Verde</strong></td>
              <td style="padding:10px 12px;text-align:right;font-family:var(--font-mono);font-weight:700;">40.00</td>
              <td style="padding:10px 12px;">Producción piscícola artesanal en jaulas flotantes sobre el Río Manacacías (20 pescadores).</td>
            </tr>
            <tr>
              <td style="padding:10px 12px;"><strong style="color:var(--col-pesca);">P03</strong></td>
              <td style="padding:10px 12px;"><strong>Asovida (San Miguel)</strong></td>
              <td style="padding:10px 12px;text-align:right;font-family:var(--font-mono);font-weight:700;">25.00</td>
              <td style="padding:10px 12px;">Captura artesanal en la Vereda San Miguel sobre el Río Meta (11 pescadores).</td>
            </tr>
            <tr>
              <td style="padding:10px 12px;"><strong style="color:var(--col-pesca);">P04</strong></td>
              <td style="padding:10px 12px;"><strong>Asopespor</strong></td>
              <td style="padding:10px 12px;text-align:right;font-family:var(--font-mono);font-weight:700;">20.00</td>
              <td style="padding:10px 12px;">Pesca artesanal en cuenca media del Río Meta (10 pescadores).</td>
            </tr>
            <tr>
              <td style="padding:10px 12px;"><strong style="color:var(--col-pesca);">P05</strong></td>
              <td style="padding:10px 12px;"><strong>Puerto Trujillo</strong></td>
              <td style="padding:10px 12px;text-align:right;font-family:var(--font-mono);font-weight:700;">30.00</td>
              <td style="padding:10px 12px;">Producción comunitaria de 40 familias en frontera fluvial (Río Iteviare).</td>
            </tr>
            <tr>
              <td style="padding:10px 12px;"><strong style="color:var(--col-pesca);">P06</strong></td>
              <td style="padding:10px 12px;"><strong>Desembarcadero Central</strong></td>
              <td style="padding:10px 12px;text-align:right;font-family:var(--font-mono);font-weight:700;">50.00</td>
              <td style="padding:10px 12px;">Punto principal de desembarque comercial urbano en Puerto Gaitán (25 pescadores).</td>
            </tr>
            <tr>
              <td style="padding:10px 12px;"><strong style="color:var(--col-pesca);">P07</strong></td>
              <td style="padding:10px 12px;"><strong>Soplaviento</strong></td>
              <td style="padding:10px 12px;text-align:right;font-family:var(--font-mono);font-weight:700;">35.00</td>
              <td style="padding:10px 12px;">Zona de ensenada y estación experimental en el Río Manacacías (8 pescadores).</td>
            </tr>
            <tr style="border-top:2px solid var(--color-saffron-spark);background:rgba(255,184,41,0.08);">
              <td style="padding:11px 12px;"><strong style="color:var(--color-saffron-spark);">TOTAL</strong></td>
              <td style="padding:11px 12px;"><strong style="color:var(--color-bone-white);">Red Completa</strong></td>
              <td style="padding:11px 12px;text-align:right;font-family:var(--font-mono);font-size:16px;font-weight:800;color:var(--color-saffron-spark);">265.75</td>
              <td style="padding:11px 12px;"><strong style="color:var(--color-warm-cream);">Carga total inyectada al sistema logístico</strong></td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="agy-prop-group" style="margin-top:16px;">
        <h4><span>Fuentes Oficiales de Verificación</span><span>Trazabilidad del Dato</span></h4>
        <div class="agy-prop-row"><span class="agy-prop-key">1. Autoridad Nacional de Acuicultura y Pesca</span><span class="agy-prop-val">Registro pesquero de la AUNAP y diagnóstico municipal de Puerto Gaitán</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">2. Trabajo de Campo Universitario y Municipal</span><span class="agy-prop-val">Censos y talleres diagnósticos de la ECA (Proyecto C05-01-2026-003 — UNILLANOS / Alcaldía)</span></div>
      </div>
    `
  }},
  kpi_merma: {{
    tag: "INDICADOR MAESTRO 2 · PÉRDIDA POSCAPTURA SIN CADENA DE FRÍO (40.0%)",
    titulo: "Fundamentación Matemática, Cinética y Algorítmica de la Merma Poscaptura",
    cuerpo: `
      <div style="display:flex;flex-direction:column;gap:14px;">
        <div class="agy-prop-group" style="margin:0;">
          <h4><span>1. El 40.0% de Pérdida Poscaptura</span><span>Worst-Case Path Analysis</span></h4>
          <p style="font-size:13.5px;margin:6px 0;">Es el porcentaje máximo de pescado que se echa a perder por completo si no existe cadena de frío.</p>
          <div class="agy-prop-row"><span class="agy-prop-key">Modelo / Teoría</span><span class="agy-prop-val">Modelo Empírico de Pérdidas Físicas y Calidad en Pesquerías Artesanales Tropicales de la FAO (<em>FAO Fisheries Technical Paper 348</em>).</span></div>
          <div class="agy-prop-row"><span class="agy-prop-key">Algoritmo / Enfoque computacional</span><span class="agy-prop-val">Evaluación de Peor Caso en Redes de Transporte (<em>Worst-Case Path Analysis</em>) en grafos dirigidos.</span></div>
          <div class="agy-prop-row"><span class="agy-prop-key">Estándar FAO</span><span class="agy-prop-val">La FAO establece que en pesquerías tropicales sin hielo se pierde entre el <strong>30% y el 40%</strong> del producto por putrefacción e histamina.</span></div>
          <div class="agy-prop-row"><span class="agy-prop-key">Caso crítico del proyecto</span><span class="agy-prop-val" style="color:var(--color-ember-accent);">Corresponde a la asociación más lejana (<strong>P05 Puerto Trujillo a 65 km</strong>), cuyo viaje sin refrigeración supera las 9.5 horas y roza exactamente ese 40% de pérdida individual.</span></div>
        </div>

        <div class="agy-prop-group" style="margin:0;">
          <h4><span>2. La Pérdida de 2.5%/h a Temperatura Ambiente (30 °C)</span><span>Cinética de Arrhenius</span></h4>
          <p style="font-size:13.5px;margin:6px 0;">Por cada hora que el pescado pasa sin hielo bajo el calor de Puerto Gaitán (30 °C), se descompone y pierde el <strong>2.5%</strong> de su masa muscular útil.</p>
          <div class="agy-prop-row"><span class="agy-prop-key">Modelo / Teoría</span><span class="agy-prop-val">Modelo de Cinética Enzimática y Velocidad Relativa de Deterioro de Spencer &amp; Baines / Regla de Arrhenius aplicada a Alimentos (FAO Doc. Técnico 348, Cap. 6.1).</span></div>
          <div class="agy-prop-row"><span class="agy-prop-key">Algoritmo / Función en código</span><span class="agy-prop-val"><code>calcular_deterioro()</code> en <code>src/utils/geo_utils.py</code> y <code>TASA_DETERIORO_BASE = 2.5</code> en <code>constants.py</code>.</span></div>
          <div class="agy-tex-block">
            <span class="agy-tex">
              <var>f</var><sub>temp</sub> <span class="agy-op">=</span> 1.0 <span class="agy-op">+</span>
              <span class="agy-frac"><span class="agy-frac-num"><var>T</var><sub>ambiente</sub></span><span class="agy-frac-den">10.0</span></span>
              <span class="agy-op">=</span> 1.0 <span class="agy-op">+</span>
              <span class="agy-frac"><span class="agy-frac-num">30.0</span><span class="agy-frac-den">10.0</span></span>
              <span class="agy-op">=</span> 4.0
            </span>
            <span class="agy-tex">
              <var>D</var>(<var>t</var>) <span class="agy-op">=</span> <var>k</var><sub>base</sub> <span class="agy-op">&times;</span> <var>t</var> <span class="agy-op">&times;</span>
              <span>(</span><span class="agy-frac"><span class="agy-frac-num"><var>f</var><sub>temp</sub></span><span class="agy-frac-den">4.0</span></span><span>)</span>
              <span class="agy-op">=</span> 2.5 <span class="agy-op">&times;</span> <var>t</var> <span class="agy-op">&times;</span>
              <span>(</span><span class="agy-frac"><span class="agy-frac-num">4.0</span><span class="agy-frac-den">4.0</span></span><span>)</span>
              <span class="agy-op">=</span> 2.5% <span class="agy-op">&times;</span> <var>t</var>
            </span>
          </div>
        </div>

        <div class="agy-prop-group" style="margin:0;">
          <h4><span>3. La Pérdida de 3.75%/h en Zonas D y qué son las Zonas D</span><span>Cuellos de Botella</span></h4>
          <p style="font-size:13.5px;margin:6px 0;">Las Zonas D son <strong>Zonas de Deterioro Crítico</strong> (<code>D01</code> Paso Río Iteviare y <code>D02</code> Cruce La Culebra): puntos geográficos descubiertos donde las embarcaciones o vehículos sufren estancamiento bajo el sol directo sin sombra ni ventilación.</p>
          <div class="agy-prop-row"><span class="agy-prop-key">Modelo / Teoría</span><span class="agy-prop-val">Modelo de Redes de Transporte con Aristas Penalizadas / Cuellos de Botella Geográficos (<em>Bottleneck Edge Weighting Model</em>).</span></div>
          <div class="agy-prop-row"><span class="agy-prop-key">Algoritmo / Función en código</span><span class="agy-prop-val">Modificador de peso de arista en <code>SimuladorCadenaFrio.simular_ruta()</code> en <code>src/optimization/cold_chain.py</code> (<code>FACTOR_DETERIORO_ZONA_D = 1.5</code>).</span></div>
          <div class="agy-tex-block">
            <span class="agy-tex">
              <var>k</var><sub>zona D</sub> <span class="agy-op">=</span> <var>k</var><sub>base</sub> <span class="agy-op">&times;</span> <var>&phi;</var><sub><i>D</i></sub> <span class="agy-op">=</span> 2.5% / h <span class="agy-op">&times;</span> 1.5 <span class="agy-op">=</span> 3.75% / h
            </span>
          </div>
        </div>

        <div class="agy-prop-group" style="margin:0;">
          <h4><span>4. Hacia Bogotá se pierde el 36.1% de la carga (95.9 kg/día)</span><span>Integración Discreta en DAG</span></h4>
          <p style="font-size:13.5px;margin:6px 0;">Es el porcentaje global de pérdida de toda la red si se intenta llevar la pesca a Bogotá sin camión refrigerado ni centro de acopio con frío:</p>
          <div class="agy-prop-row"><span class="agy-prop-key">Modelo / Teoría</span><span class="agy-prop-val">Ecuación Diferencial Discreta de Decaimiento de Masa / Modelo de Red Multimodal en Grafos Dirigidos Acíclicos (DAG).</span></div>
          <div class="agy-prop-row"><span class="agy-prop-key">Algoritmo / Función en código</span><span class="agy-prop-val"><code>SimuladorCadenaFrio.simular_ruta()</code> integrando paso a paso sobre el camino mínimo de Dijkstra.</span></div>
          <div class="agy-tex-block">
            <span class="agy-tex">
              <var>&Delta;Q</var><sub><i>i</i></sub> <span class="agy-op">=</span> <var>Q</var><sub>actual</sub> <span class="agy-op">&times;</span>
              <span>(</span><span class="agy-frac"><span class="agy-frac-num"><var>D</var><sub>tramo, <i>i</i></sub></span><span class="agy-frac-den">100</span></span><span>)</span>
              &nbsp;&nbsp;&rarr;&nbsp;&nbsp;
              <var>Q</var><sub>actual</sub> <span class="agy-op">&larr;</span> <var>Q</var><sub>actual</sub> <span class="agy-op">&minus;</span> <var>&Delta;Q</var><sub><i>i</i></sub>
            </span>
            <span class="agy-tex">
              <var>P</var><sub>Bogotá</sub> <span class="agy-op">=</span>
              <span class="agy-frac"><span class="agy-frac-num">95.9 kg perdidos</span><span class="agy-frac-den">265.75 kg capturados</span></span>
              <span class="agy-op">&times;</span> 100% <span class="agy-op">=</span> 36.1%
            </span>
          </div>
        </div>

        <div class="agy-prop-group" style="margin:0;">
          <h4><span>5. Los $1,151,000 COP/día en merma ($34.5M COP/mes)</span><span>Valoración Económica</span></h4>
          <p style="font-size:13.5px;margin:6px 0;">Es el dinero exacto que los pescadores pierden todos los días por el pescado que se les pudre en el camino.</p>
          <div class="agy-prop-row"><span class="agy-prop-key">Modelo / Teoría</span><span class="agy-prop-val">Modelo de Valoración Económica de Costos de Pérdida en Cadena de Suministro (<em>Supply Chain Spoilage Cost Model</em>).</span></div>
          <div class="agy-prop-row"><span class="agy-prop-key">Algoritmo / Función en código</span><span class="agy-prop-val">Evaluación de función de costo objetivo <span class="agy-tex"><var>C</var><sub>merma</sub> <span class="agy-op">=</span> <span class="agy-sum">&Sigma;</span> (<var>&Delta;Q</var> <span class="agy-op">&times;</span> <var>c</var><sub>unit</sub>)</span> en <code>cold_chain.py</code> (Línea 193).</span></div>
          <div class="agy-tex-block">
            <span class="agy-tex">
              <var>C</var><sub>merma</sub> <span class="agy-op">=</span> <span class="agy-sum">&Sigma;</span> (<var>&Delta;Q</var> <span class="agy-op">&times;</span> <var>c</var><sub>unit</sub>) <span class="agy-op">=</span> 95.9 kg/día <span class="agy-op">&times;</span> $12,000 COP/kg <span class="agy-op">=</span> $1,150,800 COP/día <span class="agy-op">&approx;</span> $1,151,000 COP/día
            </span>
          </div>
        </div>
      </div>
    `
  }},
  kpi_fao: {{
    tag: "INDICADOR MAESTRO 3 · LÍMITE SANITARIO FAO DE 4.0 h (240 min) — RIESGO HISTAMINA",
    titulo: "Directrices Codex Alimentarius (FAO/OMS) y Algoritmo Dijkstra con Poda TCSPP",
    cuerpo: `
      <p>Es el <strong>tiempo máximo absoluto</strong> que puede transcurrir desde que el pez sale del agua hasta que su temperatura interna baja a <strong>0 °C</strong> (mediante hielo 1:1 o centro frigorífico).</p>
      <div class="agy-prop-group" style="margin-top:14px;">
        <h4><span>Fundamentación Científica, Normativa y Computacional</span><span>Codex CAC/RCP 52-2003</span></h4>
        <div class="agy-prop-row"><span class="agy-prop-key">Modelo / Teoría</span><span class="agy-prop-val">Directrices de Inocuidad y Microbiología de Alimentos Marinos del <em>Codex Alimentarius</em> (FAO/OMS) y la FDA (<em>Code of Practice for Fish and Fishery Products, CAC/RCP 52-2003</em>).</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">Algoritmo / Enfoque computacional</span><span class="agy-prop-val">Algoritmo de Camino Más Corto Restringido por Tiempo (<em>Time-Constrained Shortest Path Problem — TCSPP</em>) implementado sobre Dijkstra con poda de aristas en tiempo real (<span class="agy-tex"><var>t</var><sub>sin frío</sub> <span class="agy-op">&le;</span> 4.0 h</span>) en <code>dijkstra_con_cota_inocuidad()</code>.</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">Fundamento biológico</span><span class="agy-prop-val">A 30 °C, las bacterias entéricas (<em>Morganella morganii</em>, <em>Klebsiella pneumoniae</em>) activan la enzima <strong>histidina descarboxilasa</strong>, convirtiendo la histidina del músculo de la cachama en <strong>histamina</strong>. Causa intoxicación escombroide grave y es <strong>termoestable</strong> (no se destruye cocinando ni congelando después). Superar 4.0 h convierte el lote en residuo biológico tóxico.</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">Cifra en la línea base</span><span class="agy-prop-val" style="color:var(--color-ember-accent);">Sin centros de frío en Puerto Gaitán, el transporte hacia mercados mayoristas alcanza hasta <strong>9.48 h (569 minutos)</strong>, superando el límite de la FAO en un <strong>237%</strong> y causando un incumplimiento sanitario del 100% en rutas largas.</span></div>
      </div>
    `
  }},
  kpi_aislamiento: {{
    tag: "INDICADOR MAESTRO 4 · LOS 63.0 km DE AISLAMIENTO FLUVIAL CRÍTICO (P05 ITEVIARE)",
    titulo: "Desglose Multimodal Tramo a Tramo de la Comunidad Más Vulnerable (40 Familias)",
    cuerpo: `
      <p>Es la distancia física y cuello de botella geográfico más severo de toda la red pesquera de Puerto Gaitán, correspondiente a <strong>P05 Puerto Trujillo</strong> (40 familias de pescadores en frontera fluvial sobre el Río Iteviare).</p>
      <div class="agy-prop-group" style="margin-top:12px;">
        <h4><span>Modelo Geodésico y Algoritmo de Optimización</span><span>WGS 84 + Haversine</span></h4>
        <div class="agy-prop-row"><span class="agy-prop-key">Modelo / Teoría</span><span class="agy-prop-val">Modelo de Red Multimodal Fluvial-Terrestre Georreferenciada (WGS 84) con penalizaciones por cambio de modo y topología de cuencas aisladas.</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">Algoritmo / Enfoque computacional</span><span class="agy-prop-val">Algoritmo de Dijkstra Multicriterio con Distancia Geodésica de Haversine y penalización cinemática por transbordo.</span></div>
      </div>

      <div style="overflow-x:auto;margin-top:14px;">
        <table class="agy-table" style="width:100%;border-collapse:collapse;">
          <thead>
            <tr>
              <th style="text-align:left;padding:9px 12px;">Tramo Dirigido (<span class="agy-tex"><var>u</var> <span class="agy-op">&rarr;</span> <var>v</var></span>)</th>
              <th style="text-align:left;padding:9px 12px;">Modo / Tipo de Vía</th>
              <th style="text-align:right;padding:9px 12px;">Distancia</th>
              <th style="text-align:right;padding:9px 12px;">Tiempo Puro</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:9px 12px;"><strong>P05 (Puerto Trujillo) &rarr; A03 (Acopio Iteviare)</strong></td>
              <td style="padding:9px 12px;color:#2997ff;">Fluvial (Río Iteviare)</td>
              <td style="padding:9px 12px;text-align:right;font-family:var(--font-mono);">3.0 km</td>
              <td style="padding:9px 12px;text-align:right;font-family:var(--font-mono);">0.20 h</td>
            </tr>
            <tr>
              <td style="padding:9px 12px;"><strong>A03 &rarr; D01 (Paso Crítico Río Iteviare)</strong></td>
              <td style="padding:9px 12px;color:var(--color-saffron-spark);">Terrestre destapado</td>
              <td style="padding:9px 12px;text-align:right;font-family:var(--font-mono);">30.0 km</td>
              <td style="padding:9px 12px;text-align:right;font-family:var(--font-mono);">1.20 h</td>
            </tr>
            <tr>
              <td style="padding:9px 12px;"><strong>D01 &rarr; M02 (Muelle Manacacías Sur)</strong></td>
              <td style="padding:9px 12px;color:var(--color-ember-accent);">Trocha mala (&times;1.5 deterioro)</td>
              <td style="padding:9px 12px;text-align:right;font-family:var(--font-mono);">8.0 km</td>
              <td style="padding:9px 12px;text-align:right;font-family:var(--font-mono);">0.40 h</td>
            </tr>
            <tr>
              <td style="padding:9px 12px;"><strong>M02 &rarr; A01 (Acopio San Miguel)</strong></td>
              <td style="padding:9px 12px;">Vía veredal (+0.25 h transbordo)</td>
              <td style="padding:9px 12px;text-align:right;font-family:var(--font-mono);">6.0 km</td>
              <td style="padding:9px 12px;text-align:right;font-family:var(--font-mono);">0.25 h</td>
            </tr>
            <tr>
              <td style="padding:9px 12px;"><strong>A01 &rarr; F02 (Centro Frío Puerto Gaitán)</strong></td>
              <td style="padding:9px 12px;color:#10b981;">Carretera secundaria</td>
              <td style="padding:9px 12px;text-align:right;font-family:var(--font-mono);">15.0 km</td>
              <td style="padding:9px 12px;text-align:right;font-family:var(--font-mono);">0.60 h</td>
            </tr>
            <tr>
              <td style="padding:9px 12px;"><strong>F02 &rarr; V_PG1 (Llanofish Puerto Gaitán)</strong></td>
              <td style="padding:9px 12px;color:#10b981;">Casco urbano pavimentado</td>
              <td style="padding:9px 12px;text-align:right;font-family:var(--font-mono);">1.0 km</td>
              <td style="padding:9px 12px;text-align:right;font-family:var(--font-mono);">0.05 h</td>
            </tr>
            <tr style="border-top:2px solid var(--color-saffron-spark);background:rgba(255,184,41,0.08);">
              <td style="padding:10px 12px;"><strong>TOTAL RUTA P05 &rarr; V_PG1</strong></td>
              <td style="padding:10px 12px;"><strong>6 Segmentos Multimodales</strong></td>
              <td style="padding:10px 12px;text-align:right;font-family:var(--font-mono);font-weight:800;color:var(--color-saffron-spark);">63.0 km</td>
              <td style="padding:10px 12px;text-align:right;font-family:var(--font-mono);font-weight:800;color:var(--color-saffron-spark);">2.70 h</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="agy-tex-block">
        <span class="agy-tex">
          <var>d</var><sub>total</sub> <span class="agy-op">=</span> 3.0 <span class="agy-op">+</span> 30.0 <span class="agy-op">+</span> 8.0 <span class="agy-op">+</span> 6.0 <span class="agy-op">+</span> 15.0 <span class="agy-op">+</span> 1.0 <span class="agy-op">=</span> 63.0 km
        </span>
        <span class="agy-tex">
          <var>t</var><sub>mov</sub> <span class="agy-op">=</span> 0.20 <span class="agy-op">+</span> 1.20 <span class="agy-op">+</span> 0.40 <span class="agy-op">+</span> 0.25 <span class="agy-op">+</span> 0.60 <span class="agy-op">+</span> 0.05 <span class="agy-op">=</span> 2.70 h
        </span>
      </div>

      <div class="agy-prop-group" style="margin-top:14px;">
        <h4><span>Efecto de Cuello de Botella y Salvamento con Centro F02</span><span><span class="agy-tex">2.95 h <span class="agy-op">&le;</span> 4.0 h</span></span></h4>
        <div class="agy-prop-row"><span class="agy-prop-key">1. Transbordo en Muelle (M02)</span><span class="agy-prop-val">Penalización obligatoria de <span class="agy-tex">+0.25 h</span> (15 minutos) de espera al sol directo (<code>TIEMPO_TRANSBORDO_MUELLE_H</code>).</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">2. Paso por Zona Crítica (D01)</span><span class="agy-prop-val">La tasa de deterioro se multiplica por <span class="agy-tex"><var>&phi;</var><sub><i>D</i></sub> <span class="agy-op">=</span> 1.5</span> (<span class="agy-tex">3.75% / h</span>).</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">Resultado con apertura de F02</span><span class="agy-prop-val" style="color:#10b981;">El tiempo acumulado sin frío llega a <span class="agy-tex">2.95 h</span>, demostrando que la apertura de F02 salva el producto dentro del umbral sanitario (<span class="agy-tex">2.95 h <span class="agy-op">&le;</span> 4.0 h</span>).</span></div>
      </div>
    `
  }},
  prob1: {{
    tag: "PROBLEMA 1 · CINÉTICA TÉRMICA (LÍNEA BASE SIN SOLUCIÓN)",
    titulo: "Degradación Enzimática y Pérdida Poscaptura a 30 °C (Hasta 40%)",
    cuerpo: `
      <p>En la cuenca de Puerto Gaitán, la temperatura ambiente promedio es de <strong>30 °C</strong>. Sin cadena de frío, el pescado recién capturado experimenta autólisis enzimática y deterioro a una tasa base de <strong>2.5% por cada hora</strong> (y <strong>3.75%/h</strong> al cruzar las zonas críticas descubiertas <code>D01</code> Río Iteviare y <code>D02</code> Cruce La Culebra).</p>
      <div class="agy-prop-group" style="margin-top:14px;">
        <h4><span>Cifras Reales en Línea Base</span><span>Sin Solución</span></h4>
        <div class="agy-prop-row"><span class="agy-prop-key">Oferta diaria expuesta (7 asociaciones)</span><span class="agy-prop-val">265.75 kg/día (7,972.5 kg/mes)</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">Vida útil a 30 °C vs. 0 °C (FAO)</span><span class="agy-prop-val">6.0 horas (30 °C) vs. 360.0 horas (0 °C)</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">Merma hacia Bogotá sin frío</span><span class="agy-prop-val" style="color:var(--color-ember-accent);">Hasta 36.1% – 40.0% (95.9 kg/día perdidos)</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">Pérdida económica directa por merma</span><span class="agy-prop-val" style="color:var(--color-ember-accent);">Hasta $1,151,000 COP/día ($12,000 COP/kg perdido)</span></div>
      </div>
    `
  }},
  prob2: {{
    tag: "PROBLEMA 2 · INOCUIDAD Y RIESGO BIOLÓGICO (HISTAMINA)",
    titulo: "Ruptura del Límite Sanitario FAO (4.0 h = 240 min) y Toxicidad por Histamina",
    cuerpo: `
      <p>La FAO y las normas sanitarias INVIMA establecen un máximo estricto de <strong>4.0 horas (240 minutos)</strong> entre la captura y la refrigeración a 0 °C. Pasado este umbral a 30 °C, las bacterias entéricas convierten el aminoácido histidina del músculo del pez en <strong>histamina</strong> (causante de intoxicación escombroide, que no se destruye al cocinar ni al congelar después).</p>
      <div class="agy-prop-group" style="margin-top:14px;">
        <h4><span>Tiempos Reales Sin Planta de Frío en Puerto Gaitán</span><span>Riesgo Crítico</span></h4>
        <div class="agy-prop-row"><span class="agy-prop-key">Tiempo P05 (Puerto Trujillo) → Mercado Local</span><span class="agy-prop-val">2.70 h (162 min) + espera en muelle M02 (+15 min)</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">Tiempo hacia Villavicencio / Bogotá sin F02</span><span class="agy-prop-val" style="color:var(--color-ember-accent);">6.78 h a 9.48 h (407 a 569 min &gt; 240 min FAO)</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">Cumplimiento sanitario hacia Bogotá sin frío</span><span class="agy-prop-val" style="color:var(--color-ember-accent);">0.0% (0 de 7 asociaciones cumplen)</span></div>
      </div>
    `
  }},
  prob3: {{
    tag: "PROBLEMA 3 · INFRAESTRUCTURA FLUVIAL Y DÉFICIT DE HIELO 1:1",
    titulo: "Ausencia de Hielo 1:1 en Canoas y Espera al Sol en Muelles (+0.25 h)",
    cuerpo: `
      <p>Las Buenas Prácticas de Pesca (BPP) exigen aplicar <strong>1 kg de hielo en escarcha por cada 1 kg de pescado (proporción 1:1)</strong> en recipientes plásticos no porosos. En la línea base, las canoas navegan sin hielo y hacen transbordo en muelles rústicos (<code>M01</code> San Miguel, <code>M02</code> Manacacías Sur, <code>M03</code> Puerto Gaitán) donde la carga espera <strong>+0.25 h (+15 min)</strong> bajo radiación solar directa.</p>
      <div class="agy-prop-group" style="margin-top:14px;">
        <h4><span>Requerimiento Físico y Económico del Hielo 1:1</span><span>Datos Reales</span></h4>
        <div class="agy-prop-row"><span class="agy-prop-key">Hielo diario requerido (1:1)</span><span class="agy-prop-val">265.75 kg hielo/día (Masa bruta total = 531.5 kg/día)</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">Costo del insumo de hielo ($800 COP/kg)</span><span class="agy-prop-val">$212,600 COP/día ($6,378,000 COP/mes)</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">Limitación de muelles M01 y M02</span><span class="agy-prop-val">Sin energía eléctrica trifásica ni agua potable para fabricar hielo in situ</span></div>
      </div>
    `
  }},
  prob4: {{
    tag: "PROBLEMA 4 · COSTO DE INVERSIÓN Y LOCALIZACIÓN EMPÍRICA (FLP)",
    titulo: "Sobrecosto de $10,000,000 COP/mes por Abrir Plantas Redundantes",
    cuerpo: `
      <p>Existen 3 ubicaciones candidatas para centros de refrigeración (<code>F01</code> Alto Neblinas, <code>F02</code> Puerto Gaitán urbano y <code>F03</code> Puerto López), cada una con capacidad de <strong>5,000 kg</strong> y un costo fijo operativo de <strong>$5,000,000 COP/mes</strong>. Sin un modelo matemático FLP, abrir los 3 centros simultáneamente eleva el costo mensual a <strong>$23,023,272 COP/mes ($104,444 COP/kg)</strong> frente a <strong>$13,023,272 COP/mes ($59,080 COP/kg)</strong> abriendo únicamente <code>F02</code>.</p>
    `
  }},
  prob5: {{
    tag: "PROBLEMA 5 · ÉPOCA DE VEDA BIOLÓGICA (ECA C05-01-2026-003)",
    titulo: "Suspensión del 100% de la Captura Fluvial durante Abril – Junio",
    cuerpo: `
      <p>Durante los meses de <strong>abril a junio</strong> (veda reproductiva AUNAP), se prohíbe la pesca extractiva en los ríos Meta, Manacacías e Iteviare. Sin una red logística que conmute hacia <strong>piscicultura sostenible de especies nativas en tierra firme</strong> (<code>G_veda</code>: 3 granjas acuícolas que aportan 240.0 kg/día), las 7 asociaciones quedan con 0.0 kg/día de flujo formal y la infraestructura frigorífica queda inactiva.</p>
    `
  }},
  prob6: {{
    tag: "PROBLEMA 6 · HÁBITATS PROTEGIDOS, RESGUARDOS Y TRAZABILIDAD",
    titulo: "Riesgo de Invasión en H01/H02 y 0% de Trazabilidad Digital ICA/INVIMA",
    cuerpo: `
      <p>La cuenca incluye áreas de restricción ambiental y territorial como el <strong>Humedal Río Manacacías (H01)</strong> y la <strong>Zona Reserva Indígena Corocito (H02)</strong> (pueblos Sikuani). Sin filtrado algorítmico (<code>evitar_restringidos=True</code>) ni estructuras de datos de registro por lote (DAG + Tabla Hash), las rutas cruzan zonas sensibles y carecen de certificación sanitaria ante el ICA y el INVIMA.</p>
    `
  }},
  teo1: {{
    tag: "TEORÍA 1 · GRAFOS DIRIGIDOS PONDERADOS Y TABLAS HASH O(1)",
    titulo: "Por qué Lista de Adyacencia sobre Tablas Hash y no Matriz Densa",
    cuerpo: `
      <p><strong>1. ¿Qué es la teoría?</strong> Un grafo dirigido ponderado <span class="agy-tex"><var>G</var> <span class="agy-op">=</span> (<var>V</var>, <var>E</var>, <var>w</var>)</span> modela entidades como vértices <span class="agy-tex"><var>V</var></span> y relaciones asimétricas con costo como aristas dirigidas <span class="agy-tex"><var>E</var> <span class="agy-op">&sube;</span> <var>V</var> <span class="agy-op">&times;</span> <var>V</var></span>. Una <strong>Tabla Hash</strong> mapea claves (<code>id_nodo</code>) a direcciones de memoria mediante una función de dispersión en tiempo constante <span class="agy-tex"><var>O</var>(1)</span>.</p>
      <p><strong>2. ¿Cómo lo implementa el programa?</strong> En <code>src/models/grafo_logistico.py</code>, <code>GrafoLogistico</code> utiliza dos diccionarios Hash nativos: <code>_nodos: Dict[str, Nodo]</code> y <code>_adyacencia: Dict[str, List[Arista]]</code>.</p>
      <p><strong>3. Defensa ante el jurado (¿Por qué así y no con Matriz 25×25?):</strong> En nuestro grafo hay <span class="agy-tex">25</span> nodos (<span class="agy-tex">25 <span class="agy-op">&times;</span> 24 <span class="agy-op">=</span> 600</span> pares posibles) pero solo <span class="agy-tex">41</span> aristas reales (<strong>densidad del 6.8%</strong>), y en <code>RedFluvial</code> hay <span class="agy-tex">12,413</span> vértices (una matriz ocuparía <span class="agy-tex">1.2 GB</span> de RAM con <span class="agy-tex">99.98%</span> de infinitos). Con Lista de Adyacencia Hash recorremos únicamente los vecinos reales en <span class="agy-tex"><var>O</var>(grado(<var>u</var>))</span> y usamos menos de <span class="agy-tex">50 KB</span>.</p>
    `
  }},
  teo2: {{
    tag: "TEORÍA 2 · MONTÍCULO BINARIO (MIN-HEAP) Y PODA BRANCH & BOUND",
    titulo: "Algoritmo de Dijkstra O((V+E) log V) y Poda Dura de Inocuidad FAO",
    cuerpo: `
      <p><strong>1. ¿Qué es la teoría?</strong> Un <strong>Montículo Binario (Min-Heap)</strong> es un árbol binario completo almacenado en un arreglo contiguo donde la clave de cada padre es menor o igual a la de sus hijos, permitiendo extraer el mínimo en <span class="agy-tex"><var>O</var>(log <var>V</var>)</span>. Combinado con una estrategia <strong>Voraz (Greedy)</strong> y <strong>Ramificación y Poda (Branch &amp; Bound)</strong>, expande siempre el camino parcial más prometedor y descarta ramas inviables.</p>
      <p><strong>2. ¿Cómo lo aplica el programa?</strong> En <code>src/algorithms/dijkstra.py</code>, <code>dijkstra()</code> halla la ruta óptima multicriterio y <code>dijkstra_con_cota_inocuidad()</code> lleva en la tupla del heap el tiempo acumulado sin frío: si al avanzar hacia un vecino <span class="agy-tex"><var>v</var></span> el tiempo sin refrigeración supera <strong>4.0 h (240 min)</strong>, poda la rama de inmediato previniendo el riesgo de histamina.</p>
    `
  }},
  teo3: {{
    tag: "TEORÍA 3 · PROGRAMACIÓN DINÁMICA EN GRAFOS",
    titulo: "Bellman-Ford O(V·E) y Floyd-Warshall O(V³)",
    cuerpo: `
      <p><strong>1. ¿Qué es la teoría?</strong> La <strong>Programación Dinámica</strong> resuelve problemas mediante el principio de optimalidad de Bellman: la solución óptima global se construye relajando inductivamente subproblemas de tamaño creciente.</p>
      <p><strong>2. Bellman-Ford (<code>src/algorithms/bellman_ford.py</code>):</strong> Relaja las <span class="agy-tex">41</span> aristas durante <span class="agy-tex">|<var>V</var>| <span class="agy-op">&minus;</span> 1 <span class="agy-op">=</span> 24</span> iteraciones sucesivas y ejecuta una iteración <span class="agy-tex">25</span> para demostrar formalmente que <code>tiene_ciclo_negativo = False</code>, coincidiendo al 100% con los tiempos de Dijkstra.</p>
      <p><strong>3. Floyd-Warshall (<code>src/algorithms/floyd_warshall.py</code>):</strong> Aplica la recurrencia triple:</p>
      <div class="agy-tex-block">
        <span class="agy-tex">
          <var>D</var><sub><i>ij</i></sub><sup>(<i>k</i>)</sup> <span class="agy-op">=</span> min<span>(</span> <var>D</var><sub><i>ij</i></sub><sup>(<i>k</i>&minus;1)</sup> , <var>D</var><sub><i>ik</i></sub><sup>(<i>k</i>&minus;1)</sup> <span class="agy-op">+</span> <var>D</var><sub><i>kj</i></sub><sup>(<i>k</i>&minus;1)</sup> <span>)</span>
        </span>
      </div>
      <p>probando cada nodo <span class="agy-tex"><var>k</var> <span class="agy-op">&in;</span> <var>V</var></span> como puente intermedio (omitiendo <span class="agy-tex"><var>k</var> <span class="agy-op">&in;</span> <var>H</var></span> cuando <code>evitar_restringidos=True</code>) para precalcular la matriz <span class="agy-tex">25 <span class="agy-op">&times;</span> 25</span> que alimenta al optimizador FLP en <span class="agy-tex"><var>O</var>(1)</span>.</p>
    `
  }},
  teo4: {{
    tag: "TEORÍA 4 · BÚSQUEDA EN AMPLITUD (BFS) Y EN PROFUNDIDAD (DFS)",
    titulo: "Exploración por Capas (Cola FIFO) y Backtracking Exhaustivo",
    cuerpo: `
      <p><strong>1. BFS con Cola FIFO (<code>collections.deque</code> en <span class="agy-tex"><var>O</var>(<var>V</var> + <var>E</var>)</span>):</strong> Explora el grafo por capas concéntricas de distancia topológica. En el proyecto determina el camino con <strong>menor número de transbordos físicos</strong> desde cada asociación hasta <code>F02</code> (evitando manipulación excesiva del pescado).</p>
      <p><strong>2. DFS con Retroceso / Backtracking:</strong> Explora profundamente cada rama marcando nodos en el camino actual y desmarcándolos al retroceder, enumerando todas las rutas simples posibles hacia el centro de frío para contar con rutas de contingencia ante derrumbes o cierres viales.</p>
    `
  }},
  teo5: {{
    tag: "TEORÍA 5 · GRAFOS ACÍCLICOS DIRIGIDOS (DAG) Y GRAFOS BITEMPORALES",
    titulo: "Orden Topológico de Kahn O(V+E) y Conmutación G_pesca ↔ G_veda",
    cuerpo: `
      <p><strong>1. DAG y Algoritmo de Kahn (<code>src/optimization/eca_solutions.py</code>):</strong> Cada lote pesquero (<code>ECA-2026-P01..P07</code>) se modela en <code>LoteTrazabilidadDAG</code>. El algoritmo de Kahn calcula los grados de entrada, encola en una FIFO los eventos sin dependencias previas y verifica que el historial sanitario sea 100% acíclico e inmutable para auditoría ICA/INVIMA.</p>
      <p><strong>2. Grafo Dependiente del Tiempo <span class="agy-tex"><var>G</var>(<var>t</var>)</span>:</strong> Modela el cambio estacional entre pesca fluvial (julio–marzo) y veda (abril–junio), conmutando los nodos fuente hacia las 3 granjas de piscicultura nativa (<code>P_PISC1..3</code>).</p>
    `
  }},
  teo6: {{
    tag: "TEORÍA 6 · OPTIMIZACIÓN COMBINATORIA (FACILITY LOCATION PROBLEM)",
    titulo: "Selección Voraz Constructiva (Greedy FLP) y Programación Entera (MILP)",
    cuerpo: `
      <p><strong>1. ¿Qué es la teoría?</strong> El <strong>Facility Location Problem (FLP)</strong> busca el subconjunto óptimo de instalaciones <span class="agy-tex"><var>F'</var> <span class="agy-op">&sube;</span> <var>F</var></span> que minimice la función objetivo:</p>
      <div class="agy-tex-block">
        <span class="agy-tex">
          min <var>Z</var> <span class="agy-op">=</span> <span class="agy-sum">&Sigma;</span> <var>f</var><sub><i>j</i></sub> <var>y</var><sub><i>j</i></sub> <span class="agy-op">+</span> <span class="agy-sum">&Sigma;</span> <var>c</var><sub><i>ij</i></sub> <var>x</var><sub><i>ij</i></sub>
        </span>
      </div>
      <p>cumpliendo cobertura sanitaria (<span class="agy-tex"><var>t</var> <span class="agy-op">&le;</span> 4.0 h</span>) y capacidad (<span class="agy-tex">5,000 kg</span>).</p>
      <p><strong>2. Defensa del resultado:</strong> Evaluando los 5 escenarios con <code>EvaluadorEscenarios</code>, se demuestra que abrir <strong>únicamente F02 (Puerto Gaitán)</strong> cubre al 100% de las asociaciones en <span class="agy-tex"><span class="agy-op">&le;</span> 2.20 h</span> y ahorra <strong>$10,000,000 COP/mes</strong> frente a abrir las 3 plantas.</p>
    `
  }},
  externos: {{
    tag: "DELIMITACIÓN PROFESIONAL DE ALCANCE · SOFTWARE VS. DISCIPLINAS EXTERNAS",
    titulo: "Por qué estos 5 Problemas No Pueden Resolverse Mediante Código por Desarrolladores",
    cuerpo: `
      <p>Un ingeniero de sistemas debe defender con rigor dónde termina la solución computacional y dónde inicia la responsabilidad de otras ciencias:</p>
      <div class="agy-prop-group" style="margin-top:12px;">
        <h4><span>Desglose Técnico de Límites No Computacionales</span><span>Justificación</span></h4>
        <div class="agy-prop-row"><span class="agy-prop-key">1. Contaminación por Mercurio (Hg)</span><span class="agy-prop-val">Es una propiedad química atómica del tejido del pez; requiere digestión ácida y espectrometría en laboratorio (Grupo GRITOX), no un algoritmo.</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">2. Fechas biológicas de la veda</span><span class="agy-prop-val">El desove depende del pulso hidrológico del río Meta y hormonas reproductivas (Biología AUNAP/SUSA). El software recibe las fechas como parámetro y adapta la red.</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">3. Inspección sensorial de branquias y ojos</span><span class="agy-prop-val">Verificar en vivo el olor, mucosidad y color rojo de las agallas al descargar la canoa requiere inspección física veterinaria/bromatológica (INVIMA).</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">4. Sustitución de madera y obras en muelles</span><span class="agy-prop-val">Cambiar mesas de madera por acero inoxidable y construir techos/acueducto en M01–M03 requiere presupuesto físico e Ingeniería Civil (Alcaldía).</span></div>
        <div class="agy-prop-row"><span class="agy-prop-key">5. Gobernanza y consulta previa indígena</span><span class="agy-prop-val">El software bloquea el paso por H02 en el grafo, pero los acuerdos de co-manejo con los resguardos Sikuani son procesos jurídicos y comunitarios.</span></div>
      </div>
    `
  }}
}};

function agyAbrirModalDetalle(key) {{
  const info = AGY_MODALES_INFO[key];
  const backdrop = document.getElementById("agy-modal-backdrop");
  const content = document.getElementById("agy-modal-content");
  if (!info || !backdrop || !content) return;
  content.innerHTML = `
    <span class="agy-badge" style="color:var(--color-saffron-spark);margin-bottom:10px;">${{info.tag}}</span>
    <h2 id="agy-modal-title" style="margin-bottom:14px;">${{info.titulo}}</h2>
    <div style="font-size:15px;line-height:1.65;color:var(--color-silver-mist);">${{info.cuerpo}}</div>
  `;
  backdrop.classList.add("open");
}}

function agyAbrirModalConstruccionSubgrafo() {{
  const backdrop = document.getElementById("agy-modal-backdrop");
  const content = document.getElementById("agy-modal-content");
  if (!backdrop || !content) return;

  let titulo = "Construcción y Defensa del Grafo Mixto Multimodal G=(V,E)";
  let tag = "DIAPOSITIVA 5 · UNIÓN DEL GRAFO VIAL + GRAFO DE RÍOS";
  let html = `
    <p><strong>¿Cómo se unieron los dos grafos y por qué se decidió así?</strong></p>
    <p>1. <strong>Creación de la estructura base:</strong> Primero se creó <code>GrafoLogistico</code> con listas de adyacencia Hash <code>O(1)</code> y se tipificaron los <strong>25 vértices en 7 roles reales (P, M, A, F, D, H, V)</strong>.<br>
    2. <strong>Soldadura multimodal en Muelles (M) y Acopios (A):</strong> Las <strong>10 aristas fluviales</strong> (calculadas sobre los 12,413 vértices de <code>rios.shp</code>) desembocan en los nodos puente <code>M01, M02, M03</code> y <code>A01, A02, A03</code>, desde donde nacen las <strong>31 aristas terrestres</strong> (12 de acceso local + 19 carreteras OSRM).<br>
    3. <strong>Analogía de ingeniería:</strong> Opera como una red de transmisión donde la frescura se atenúa a 2.5%/h en el río, sufre +0.25 h de latencia en los muelles convertidores (M), se multiplica ×1.5 en resistencias térmicas (D), evita cortafuegos ambientales (H) y se regenera a 0 °C en el repetidor frigorífico (F02).</p>
  `;
  if (agySlideActual === 2) {{
    tag = "DIAPOSITIVA 3 · METODOLOGÍA DE CONSTRUCCIÓN DEL SUBGRAFO VIAL";
    titulo = "Cómo se Construyó el Grafo Vial (19 Vértices · 31 Aristas Terrestres)";
    html = `
      <p><strong>Orden exacto de construcción en código:</strong></p>
      <p>1.º Se filtraron los 19 vértices con conexión terrestre (Muelles <code>M</code>, Acopios <code>A</code>, Centros de Frío <code>F</code>, Zonas <code>D/H</code> y Mercados <code>V</code>).<br>
      2.º Se cargaron de <code>aristas.csv</code> las <strong>31 conexiones viales</strong> separadas en dos categorías físicas: <strong>12 vías de <code>acceso_local</code></strong> (trochas veredales de baja velocidad que conectan los ríos con la carretera) y <strong>19 vías <code>terrestres</code></strong> (corredor pavimentado Ruta Nacional 40 hacia Puerto López, Villavicencio y Bogotá).<br>
      3.º Con <code>route_geometry.py</code> se consultó el motor <strong>OSRM</strong> y se anclaron los extremos exactos con <code>_anclar_extremos()</code>, guardando las polilíneas reales en <code>rutas_cache.json</code> para ejecución 100% offline.</p>
    `;
  }} else if (agySlideActual === 3) {{
    tag = "DIAPOSITIVA 4 · METODOLOGÍA DE CONSTRUCCIÓN DEL GRAFO DE RÍOS";
    titulo = "Cómo se Construyó el Grafo Fluvial (12,413 Vértices IGAC → 10 Rutas)";
    html = `
      <p><strong>Orden exacto de construcción en <code>src/visualization/river_network.py</code>:</strong></p>
      <p>1.º Se cargó el shapefile oficial <code>data/shapefiles/rios.shp</code> (IGAC) recortado a la cuenca de los ríos Meta, Manacacías e Iteviare.<br>
      2.º Cada cauce se discretizó en <strong>12,413 vértices <code>(lon, lat)</code></strong> (redondeados a 5 decimales ≈ 1.1 m) y <strong>24,790 segmentos ponderados con distancia geodésica de Haversine</strong> en un grafo no dirigido (<code>RedFluvial</code>).<br>
      3.º Se indexó el territorio en una <strong>Rejilla Espacial Hash de 447 celdas (0.05°×0.05°)</strong> para proyectar cada asociación pesquera (<code>P01..P07</code>) al cauce más cercano en <span class="agy-tex"><var>O</var>(1)</span> y trazar con Dijkstra propio las <strong>10 rutas fluviales reales</strong>.</p>
    `;
  }}
  content.innerHTML = `
    <span class="agy-badge" style="color:var(--color-saffron-spark);margin-bottom:10px;">${{tag}}</span>
    <h2 id="agy-modal-title" style="margin-bottom:14px;">${{titulo}}</h2>
    <div style="font-size:15px;line-height:1.65;color:var(--color-silver-mist);">${{html}}</div>
  `;
  backdrop.classList.add("open");
}}

function agyCerrarModalDetalle() {{
  document.getElementById("agy-modal-backdrop")?.classList.remove("open");
}}

const AGY_VARS_DOCS = {{
  nodo: {{
    titulo: "Anatomía del Vértice — 12 Variables Reales (src/models/nodo.py)",
    secciones: [
      {{
        id: 1,
        label: "1. Identidad y GPS",
        intro: "Identifican cada punto físico del territorio (pesca, muelle, acopio, frío, zona crítica, hábitat o mercado), fijan su posición en el elipsoide WGS 84 y determinan su rol biológico y algorítmico dentro del grafo.",
        vars: [
          {{
            name: "id", type: "str", origen: "Dato CSV", where: "grafo_logistico.py · dijkstra.py",
            desc: "Clave única del vértice (ej. P01, M01, F02). Indexa el nodo dentro de la tabla hash de adyacencia para recuperarlo en tiempo constante.",
            live: (n) => n ? n.id : null,
            formulaCap: "Acceso en memoria (Tabla Hash)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><var>T</var>(<code>obtener_nodo(id)</code>)<span class="agy-op">=</span><var>O</var>(1)</div></div>'
          }},
          {{
            name: "nombre", type: "str", origen: "Dato CSV", where: "folium_map.py · graph_slides.py",
            desc: "Denominación real del lugar o asociación (ej. ASOPESGA, Muelle Río Meta Alto). Proporciona trazabilidad legible en mapas, tablas e informes.",
            live: (n) => n ? n.nombre : null
          }},
          {{
            name: "tipo", type: "str", origen: "Dato CSV", where: "cold_chain.py (L148–178) · nodo.py",
            desc: "Código de rol (P, M, A, F, D, H, V). Actúa como conmutador físico-biológico al atravesar el vértice durante la simulación térmica y el ruteo.",
            live: (n) => n ? `${{n.tipo}} · ${{n.tipo_descripcion}}` : null,
            formulaCap: "Reglas de transición por tipo de nodo (cold_chain.py)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><b>D:</b>&nbsp;<var>D</var><sub>tramo</sub><span class="agy-op">&larr;</span><var>D</var><sub>tramo</sub><span class="agy-op">&middot;</span>1.5<span class="agy-vm-math-sep"></span><b>M:</b>&nbsp;<var>t</var><sub>tramo</sub><span class="agy-op">&larr;</span><var>t</var><sub>tramo</sub><span class="agy-op">+</span>0.25 h</div><div class="agy-vm-math-row"><b>F / A</b>&nbsp;(<var>C</var><sub>fr&iacute;o</sub><span class="agy-op">&gt;</span>0):&nbsp;<var>t</var><sub>sin fr&iacute;o</sub><span class="agy-op">&larr;</span>0 h,&nbsp;&nbsp;<var>T</var><span class="agy-op">&larr;</span>0 &deg;C</div></div>'
          }},
          {{
            name: "latitud / longitud", type: "float", origen: "Dato GPS (WGS 84)", where: "geo_utils.py (L18–43) · route_geometry.py",
            desc: "Coordenadas geográficas decimales Norte y Oeste. Permiten trazar geometrías reales sobre OSRM/IGAC y calcular distancias geodésicas esféricas mediante Haversine.",
            live: (n) => n ? `${{n.latitud.toFixed(4)}}°N, ${{n.longitud.toFixed(4)}}°W` : null,
            formulaCap: "Cálculo de distancia geodésica — Fórmula de Haversine (R = 6371 km)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><var>a</var><span class="agy-op">=</span>sin<sup>2</sup><span class="agy-frac"><span class="agy-frac-num">&Delta;&phi;</span><span class="agy-frac-den">2</span></span><span class="agy-op">+</span>cos(<var>&phi;</var><sub>1</sub>)&middot;cos(<var>&phi;</var><sub>2</sub>)&middot;sin<sup>2</sup><span class="agy-frac"><span class="agy-frac-num">&Delta;&lambda;</span><span class="agy-frac-den">2</span></span></div><div class="agy-vm-math-row"><var>d</var><span class="agy-op">=</span>2<var>R</var><span class="agy-op">&middot;</span>atan2(&radic;<var>a</var>,&nbsp;&radic;(1&minus;<var>a</var>))</div></div>'
          }}
        ]
      }},
      {{
        id: 2,
        label: "2. Producción y Costos",
        intro: "Cuantifican la biomasa diaria inyectada en origen, los límites de almacenamiento frigorífico y la inversión mensual fija que gobiernan el Problema de Localización de Instalaciones (FLP).",
        vars: [
          {{
            name: "produccion_kg_dia", type: "float", origen: "Dato AUNAP / ECA", where: "grafo_logistico.py · cold_chain.py · eca_solutions.py",
            desc: "Volumen diario de pescado fresco capturado en nodos de origen P. Define la masa inicial del simulador de merma y el requerimiento de hielo 1:1.",
            live: (n) => n ? `${{n.produccion_kg_dia.toFixed(2)}} kg/día` : null,
            formulaCap: "Carga total de la red y balance de hielo 1:1",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><var>Q</var><sub>tot</sub><span class="agy-op">=</span><span class="agy-sum">&sum;</span><sub><i>i</i>&isin;<i>P</i></sub>&nbsp;<var>q</var><sub><i>i</i></sub><span class="agy-op">=</span>265.75 kg/d&iacute;a<span class="agy-vm-math-sep"></span><var>Q</var><sub>hielo</sub><span class="agy-op">=</span>1.0<span class="agy-op">&middot;</span><var>q</var><sub><i>i</i></sub></div></div>'
          }},
          {{
            name: "capacidad_frio_kg", type: "float", origen: "Dato CSV", where: "facility_location.py · cold_chain.py (L145)",
            desc: "Tonelaje máximo que puede refrigerar un nodo F o A. Actúa como restricción dura de capacidad en el FLP y activa el reinicio térmico cuando es mayor que cero.",
            live: (n) => n ? `${{n.capacidad_frio_kg.toLocaleString()}} kg` : null,
            formulaCap: "Restricción de capacidad del modelo FLP (facility_location.py)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><span class="agy-sum">&sum;</span><sub><i>i</i>&isin;<i>P</i></sub>&nbsp;<var>q</var><sub><i>i</i></sub><span class="agy-op">&middot;</span><var>x</var><sub><i>ij</i></sub><span class="agy-op">&le;</span><var>C</var><sub><i>j</i></sub><span class="agy-op">&middot;</span><var>y</var><sub><i>j</i></sub>&nbsp;&nbsp;&nbsp;(&forall;&nbsp;<var>j</var>&nbsp;&isin;&nbsp;<var>F</var>)</div></div>'
          }},
          {{
            name: "costo_fijo", type: "float", origen: "Dato CSV (COP/mes)", where: "facility_location.py · scenario_evaluator.py",
            desc: "Costo mensual de operación, nómina y energía de abrir una planta (ej. $3,000,000 COP/mes en F02). Penaliza abrir plantas redundantes en la función objetivo FLP.",
            live: (n) => n ? `$${{n.costo_fijo.toLocaleString()}} COP/mes` : null,
            formulaCap: "Función objetivo del FLP (apertura fija + operación)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row">min&nbsp;<var>Z</var><span class="agy-op">=</span><span class="agy-sum">&sum;</span><sub><i>j</i>&isin;<i>F</i></sub>&nbsp;<var>F</var><sub><i>j</i></sub><span class="agy-op">&middot;</span><var>y</var><sub><i>j</i></sub><span class="agy-op">+</span><span class="agy-sum">&sum;</span><sub><i>i,j</i></sub>&nbsp;<var>c</var><sub><i>ij</i></sub><span class="agy-op">&middot;</span><var>x</var><sub><i>ij</i></sub></div></div>'
          }},
          {{
            name: "metadata (demanda, río, pescadores)", type: "dict", origen: "Dato CSV flexible", where: "data_loader.py · graph_slides.py",
            desc: "Diccionario extensible que guarda la demanda comercial en mercados V, el río asociado y el censo de pescadores por asociación sin alterar el esquema base de Nodo.",
            live: (n) => n ? `Demanda: ${{n.metadata.demanda_kg_dia || 0}} kg/d · Río: ${{n.metadata.rio || "—"}}` : null
          }}
        ]
      }},
      {{
        id: 3,
        label: "3. Infraestructura y Filtros",
        intro: "Variables booleanas que evalúan la viabilidad técnica real en campo (energía, agua, muelle) y aplican bloqueos ambientales estrictos dentro de los algoritmos de caminos mínimos.",
        vars: [
          {{
            name: "tiene_energia", type: "bool", origen: "Dato CSV", where: "nodo.py (L92) · facility_location.py",
            desc: "Indica si el punto cuenta con red eléctrica trifásica confiable para operar compresores y fabricar hielo. Excluye automáticamente veredas sin red (como A03).",
            live: (n) => n ? (n.tiene_energia ? "True (Energía continua)" : "False (Sin red estable)") : null
          }},
          {{
            name: "tiene_agua", type: "bool", origen: "Dato CSV", where: "nodos_centros.csv · validaciones BPP",
            desc: "Verifica disponibilidad de agua potable, requisito sanitario obligatorio de la FAO e INVIMA para fabricar hielo higiénico en escarcha y lavado de captura.",
            live: (n) => n ? (n.tiene_agua ? "True (Disponible)" : "False") : null
          }},
          {{
            name: "tiene_muelle", type: "bool", origen: "Dato CSV", where: "nodo.py · aristas.csv",
            desc: "Señala si existe infraestructura física de desembarque para atracar lanchas y transferir carga al modo terrestre sin deterioro por arrastre.",
            live: (n) => n ? (n.tiene_muelle ? "True (Con muelle)" : "False") : null
          }},
          {{
            name: "restriccion_ambiental", type: "bool", origen: "Dato CSV", where: "dijkstra.py (L18–21, L83) · bellman_ford.py",
            desc: "Marca reservas ecológicas o resguardos protegidos (H01, H02). Cuando evitar_restringidos=True está activo, Dijkstra y Bellman-Ford vetan el nodo como paso intermedio.",
            live: (n) => n ? (n.restriccion_ambiental ? "True (Hábitat vetado H)" : "False (Tránsito libre)") : null,
            formulaCap: "Regla de poda ambiental en caminos mínimos",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><code>restriccion_ambiental(v) = True</code><span class="agy-op">&rArr;</span><var>w</var>(<var>u</var>,&nbsp;<var>v</var>)<span class="agy-op">=</span>&infin;&nbsp;<span class="agy-vm-math-sub">(no transitable)</span></div></div>'
          }},
          {{
            name: "es_candidato_frio", type: "@property (bool)", origen: "Calculado en código", where: "src/models/nodo.py (L88–94)",
            desc: "Propiedad booleana derivada que filtra los 5 nodos aptos para el algoritmo FLP (F01, F02, F03, A01, A02) combinando tipo, energía y ausencia de veto ecológico.",
            live: (n) => n ? (n.es_candidato_frio ? "True (Candidato FLP válido)" : "False (No elegible)") : null,
            formulaCap: "Predicado lógico de elegibilidad FLP (nodo.py)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><var>candidato</var>(<var>v</var>)<span class="agy-op">&hArr;</span>(<var>tipo</var>&nbsp;&isin;&nbsp;{{A, F}})<span class="agy-op">&and;</span><var>energ&iacute;a</var><span class="agy-op">&and;</span>&not;<var>restricci&oacute;n</var></div></div>'
          }}
        ]
      }},
      {{
        id: 4,
        label: "4. Conectividad y Topología",
        intro: "Métricas estructurales derivadas directamente de las listas de adyacencia del grafo dirigido G=(V,E), fundamentales para detectar nodos concentradores (hubs) y puntos de aislamiento.",
        vars: [
          {{
            name: "grado_entrada / grado_salida", type: "int (calculado)", origen: "Calculado en código", where: "grafo_logistico.py · graph_slides.py",
            desc: "Número de aristas que llegan al vértice (deg⁻) y que salen desde él (deg⁺). Un nodo concentrador como F02 exhibe alto grado de entrada desde acopios/muelles y salidas hacia mercados.",
            live: (n) => n ? `deg⁻ = ${{n.grado_entrada}} entrantes · deg⁺ = ${{n.grado_salida}} salientes` : null,
            formulaCap: "Definición formal en grafo dirigido y costo espacial",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row">deg<sup>+</sup>(<var>v</var>)<span class="agy-op">=</span>|adj[<var>v</var>]|<span class="agy-vm-math-sep"></span>deg<sup>&minus;</sup>(<var>v</var>)<span class="agy-op">=</span>|{{<var>u</var>&nbsp;&isin;&nbsp;<var>V</var>&nbsp;:&nbsp;(<var>u</var>,<var>v</var>)&nbsp;&isin;&nbsp;<var>E</var>}}|</div><div class="agy-vm-math-row agy-vm-math-sub">Costo espacial de lista de adyacencia:&nbsp;<var>O</var>(|<var>V</var>|&nbsp;+&nbsp;|<var>E</var>|)</div></div>'
          }},
          {{
            name: "modos_conectados", type: "list[str]", origen: "Calculado en código", where: "graph_slides.py",
            desc: "Conjunto único de modalidades de transporte (fluvial, acceso_local, terrestre) que inciden en el vértice. Identifica automáticamente los nodos de transbordo multimodal.",
            live: (n) => n ? ((n.modos_conectados || []).join(" + ") || "—") : null
          }}
        ]
      }}
    ]
  }},
  arista: {{
    titulo: "Anatomía de la Arista — 20 Variables + 6 Fórmulas (src/models/arista.py)",
    secciones: [
      {{
        id: 0,
        label: "★ Peso Multicriterio w_ij",
        intro: "Un camión o lancha con pescado perecedero no puede evaluarse con una sola métrica: elegir solo distancia envía la carga por trochas destruidas; elegir solo tiempo dispara combustible; elegir solo costo ignora el riesgo de derrumbe. Por eso Arista.peso() sintetiza 6 criterios normalizados en [0, 1].",
        vars: [
          {{
            name: "peso() / peso_default", type: "float (calculado)", origen: "Calculado en código", where: "src/models/arista.py (L139–173)",
            desc: "Peso escalar adimensional usado por Dijkstra, Bellman-Ford y Floyd-Warshall para encontrar rutas simultáneamente rápidas, económicas, confiables y seguras.",
            live: (a) => a ? `w_ij = ${{a.peso_combinado.toFixed(4)}}` : null,
            formulaCap: "Fórmula Maestra Multicriterio de 6 Factores (arista.py)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><var>w</var><sub><i>ij</i></sub><span class="agy-op">=</span>0.25<var>t&#770;</var><sub><i>ij</i></sub><span class="agy-op">+</span>0.20<var>c&#770;</var><sub><i>ij</i></sub><span class="agy-op">+</span>0.20<var>r</var><sub>comb</sub><span class="agy-op">+</span>0.10<var>p&#770;</var><sub><i>ij</i></sub><span class="agy-op">+</span>0.10(1&minus;<var>q</var><sub><i>ij</i></sub>)<span class="agy-op">+</span>0.05<var>P</var><sub>ret</sub></div><div class="agy-vm-math-row agy-vm-math-sub"><var>t&#770;</var><sub><i>ij</i></sub><span class="agy-op">=</span><span class="agy-frac"><span class="agy-frac-num"><var>t</var><sub>efec</sub></span><span class="agy-frac-den">48</span></span><span class="agy-vm-math-sep"></span><var>c&#770;</var><sub><i>ij</i></sub><span class="agy-op">=</span><span class="agy-frac"><span class="agy-frac-num"><var>C</var><sub>tot</sub></span><span class="agy-frac-den">5,000,000</span></span><span class="agy-vm-math-sep"></span><var>p&#770;</var><sub><i>ij</i></sub><span class="agy-op">=</span><span class="agy-frac"><span class="agy-frac-num"><var>L</var><sub><i>ij</i></sub></span><span class="agy-frac-den">100</span></span></div></div>'
          }}
        ]
      }},
      {{
        id: 1,
        label: "1. Cinemática y Tiempo",
        intro: "Describen la dirección del tramo y su cinemática física real. El tiempo efectivo es la variable de mayor peso (α = 0.25) porque gobierna directamente el reloj biológico del pescado.",
        vars: [
          {{
            name: "origen / destino", type: "str", origen: "Dato CSV", where: "grafo_logistico.py · dijkstra.py",
            desc: "IDs de los vértices inicial y final que definen la dirección del arco (u → v) dentro de la lista de adyacencia del grafo dirigido.",
            live: (a) => a ? `${{a.origen}} → ${{a.destino}}` : null
          }},
          {{
            name: "distancia_km", type: "float", origen: "Dato OSRM / rios.shp", where: "aristas.csv · route_geometry.py",
            desc: "Longitud real del tramo en kilómetros obtenida sobre el trazado vial OSRM o sobre el cauce hidrográfico real de rios.shp (no línea recta).",
            live: (a) => a ? `${{a.distancia_km.toFixed(2)}} km` : null
          }},
          {{
            name: "tiempo_h", type: "float", origen: "Dato CSV / geo_utils.py", where: "dijkstra.py · cold_chain.py · geo_utils.py (L46)",
            desc: "Duración base del recorrido sin esperas. Cuando se estima automáticamente desde distancia y estado vial usa la velocidad efectiva.",
            live: (a) => a ? `${{a.tiempo_h.toFixed(2)}} h` : null,
            formulaCap: "Estimación automática de tiempo base (geo_utils.py · L68)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><var>t</var><sub>base</sub><span class="agy-op">=</span><span class="agy-frac"><span class="agy-frac-num"><var>d</var><sub>km</sub></span><span class="agy-frac-den"><var>v</var><sub>prom</sub> / <var>f</var><sub>v&iacute;a</sub></span></span></div></div>'
          }},
          {{
            name: "flujo_vehicular → factor_flujo", type: "str → @property", origen: "Dato CSV + Derivado", where: "src/models/arista.py (L100–103)",
            desc: "Nivel de congestión del corredor ('bajo', 'medio', 'alto'). Se transforma en un multiplicador de tiempo: bajo = 1.0, medio = 1.15, alto = 1.35.",
            live: (a) => a ? `${{a.flujo_vehicular}} (×${{a.factor_flujo}})` : null
          }},
          {{
            name: "tiempo_espera_h", type: "float", origen: "Dato CSV", where: "arista.py (L116)",
            desc: "Horas muertas acumuladas en filas, cargue, descargue o casetas de peaje. Se suma directamente al tiempo efectivo del tramo.",
            live: (a) => a ? `${{a.tiempo_espera_h.toFixed(2)}} h` : null
          }},
          {{
            name: "tiempo_efectivo_h / velocidad_efectiva", type: "@property (float)", origen: "Calculado en código", where: "src/models/arista.py (L114–116)",
            desc: "Tiempo real total del tramo ajustado por flujo vehicular y esperas, del cual deriva la velocidad efectiva y el término α·t̂ en el peso multicriterio.",
            live: (a) => a ? `${{a.tiempo_efectivo_h.toFixed(3)}} h (${{a.velocidad_efectiva_kmh.toFixed(1)}} km/h)` : null,
            formulaCap: "Cálculo de tiempo y velocidad efectivos (arista.py · L116)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><var>t</var><sub>efec</sub><span class="agy-op">=</span>(<var>t</var><sub>base</sub><span class="agy-op">&middot;</span><var>f</var><sub>flujo</sub>)<span class="agy-op">+</span><var>t</var><sub>espera</sub><span class="agy-vm-math-sep"></span><var>v</var><sub>efec</sub><span class="agy-op">=</span><span class="agy-frac"><span class="agy-frac-num"><var>d</var><sub>km</sub></span><span class="agy-frac-den"><var>t</var><sub>efec</sub></span></span></div></div>'
          }}
        ]
      }},
      {{
        id: 2,
        label: "2. Costos y Peajes",
        intro: "Modelan el gasto monetario directo en pesos colombianos (COP) por flete y peajes. Aportan el segundo criterio de mayor peso (β = 0.20) en la función multicriterio.",
        vars: [
          {{
            name: "costo_cop", type: "float", origen: "Dato CSV / Auto", where: "arista.py (L86–87)",
            desc: "Costo del flete y combustible del tramo en COP. Si no viene especificado en el CSV, __post_init__ lo calcula automáticamente a $3,500 COP/km.",
            live: (a) => a ? `$${{a.costo_cop.toLocaleString()}} COP` : null,
            formulaCap: "Cálculo por defecto de flete (arista.py · L87)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><var>C</var><sub>flete</sub><span class="agy-op">=</span><var>d</var><sub>km</sub><span class="agy-op">&times;</span>$3,500 COP/km</div></div>'
          }},
          {{
            name: "peajes / costo_peajes_cop", type: "int / float", origen: "Dato CSV / Auto", where: "arista.py (L88–89)",
            desc: "Número de casetas de peaje en la carretera (ej. corredor Villavicencio–Bogotá) y su importe total. Por defecto se estima a $12,000 COP por caseta.",
            live: (a) => a ? `${{a.peajes}} casetas ($${{a.costo_peajes_cop.toLocaleString()}} COP)` : null,
            formulaCap: "Estimación de peajes por defecto (arista.py · L89)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><var>C</var><sub>peajes</sub><span class="agy-op">=</span><var>n</var><sub>peajes</sub><span class="agy-op">&times;</span>$12,000 COP</div></div>'
          }},
          {{
            name: "costo_total_cop", type: "@property (float)", origen: "Calculado en código", where: "src/models/arista.py (L119–121, L157)",
            desc: "Suma íntegra del costo monetario del tramo. Se normaliza dividiendo por $5,000,000 COP para ingresar como β·ĉ_ij en el peso multicriterio.",
            live: (a) => a ? `$${{a.costo_total_cop.toLocaleString()}} COP` : null,
            formulaCap: "Costo total y término normalizado en w_ij (arista.py · L121)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><var>C</var><sub>tot</sub><span class="agy-op">=</span><var>C</var><sub>flete</sub><span class="agy-op">+</span><var>C</var><sub>peajes</sub><span class="agy-vm-math-sep"></span><var>c&#770;</var><sub><i>ij</i></sub><span class="agy-op">=</span><span class="agy-frac"><span class="agy-frac-num"><var>C</var><sub>tot</sub></span><span class="agy-frac-den">5,000,000</span></span></div></div>'
          }}
        ]
      }},
      {{
        id: 3,
        label: "3. Riesgo, Vía y Merma",
        intro: "Variables físicas y operativas extendidas (MCDM) que capturan la fragilidad de vías en el Meta, el riesgo térmico y de derrumbe, restricciones de carga y el deterioro esperado por tramo.",
        vars: [
          {{
            name: "tipo_via", type: "str", origen: "Dato CSV", where: "eca_solutions.py (veda) · folium_map.py",
            desc: "Modalidad física ('fluvial', 'acceso_local', 'terrestre'). Permite al Grafo Bitemporal desactivar automáticamente las 10 aristas fluviales durante la veda (abril–junio).",
            live: (a) => a ? a.tipo_via : null
          }},
          {{
            name: "estado_via → factor_estado_via", type: "str → @property", origen: "Dato CSV + Derivado", where: "src/models/arista.py (L94–97)",
            desc: "Condición superficial de la ruta ('buena', 'regular', 'mala'). Se traduce en el multiplicador de deterioro: buena = 1.0, regular = 1.5, mala = 2.0.",
            live: (a) => a ? `${{a.estado_via}} (×${{a.factor_estado_via}})` : null
          }},
          {{
            name: "riesgo_termico / accidente / derrumbe → riesgo_combinado", type: "float → @property", origen: "Dato CSV + Calculado", where: "src/models/arista.py (L106–111)",
            desc: "Tres riesgos normalizados [0–1]: exposición solar/calor (50%), siniestralidad vial o palizadas (25%) y caída de banca en lluvias (25%). Alimentan el término γ = 0.20.",
            live: (a) => a ? `Térm=${{a.riesgo_termico.toFixed(2)}} · Acc=${{a.riesgo_accidente.toFixed(2)}} · Derr=${{a.riesgo_derrumbe.toFixed(2)}} → r_comb=${{a.riesgo_combinado.toFixed(4)}}` : null,
            formulaCap: "Riesgo combinado del tramo (arista.py · L108)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><var>r</var><sub>comb</sub><span class="agy-op">=</span>min(0.50<var>r</var><sub>t&eacute;rm</sub><span class="agy-op">+</span>0.25<var>r</var><sub>acc</sub><span class="agy-op">+</span>0.25<var>r</var><sub>derr</sub>,&nbsp;&nbsp;1.0)</div></div>'
          }},
          {{
            name: "confiabilidad_via / prob_retraso → perdida_esperada", type: "float → @property", origen: "Dato CSV + Calculado", where: "src/models/arista.py (L124–135)",
            desc: "Confiabilidad q_ij ∈ [0,1] (penaliza incertidumbre con ε = 0.10) y probabilidad de retén/trancón P_ret (ζ = 0.05). Junto con tiempo y estado vial calculan la merma esperada L_ij (δ = 0.10).",
            live: (a) => a ? `q=${{a.confiabilidad_via.toFixed(2)}} · P_ret=${{a.prob_retraso.toFixed(2)}} → L_ij=${{a.perdida_esperada.toFixed(2)}}%` : null,
            formulaCap: "Porcentaje de pérdida esperada por tramo (arista.py · L128)",
            formula: '<div class="agy-vm-math-stack"><div class="agy-vm-math-row"><var>L</var><sub><i>ij</i></sub><span class="agy-op">=</span>min(2.5<span class="agy-op">&middot;</span><var>t</var><sub>efec</sub><span class="agy-op">&middot;</span><var>f</var><sub>v&iacute;a</sub><span class="agy-op">&middot;</span>(1<span class="agy-op">+</span>0.5(1&minus;<var>q</var><sub><i>ij</i></sub>))<span class="agy-op">&middot;</span>(1<span class="agy-op">+</span>0.3<var>P</var><sub>ret</sub>),&nbsp;&nbsp;100%)</div></div>'
          }},
          {{
            name: "capacidad_kg / restriccion_peso_ton / horario / num_transbordos", type: "float / str / int", origen: "Datos CSV operativos", where: "aristas.csv · eca_solutions.py · cold_chain.py",
            desc: "Restricciones logísticas reales: cupo máximo del vehículo (300 kg canoa a 2,000 kg camión; valida hielo 1:1), límite de tonelaje en puentes rurales, ventanas horarias y número de transbordos.",
            live: (a) => a ? `Cap: ${{a.capacidad_kg}} kg · Puente: ${{a.restriccion_peso_ton}} t · Horario: ${{a.horario_restriccion}}` : null
          }}
        ]
      }}
    ]
  }}
}};

function agyAbrirModalVariables(entidad, seccionId) {{
  const backdrop = document.getElementById("agy-modal-backdrop");
  const content = document.getElementById("agy-modal-content");
  if (!backdrop || !content) return;

  const ent = entidad === "arista" ? "arista" : "nodo";
  const cfg = AGY_VARS_DOCS[ent];
  const sec = cfg.secciones.find(s => s.id === seccionId) || cfg.secciones[0];

  // Obtener instancia seleccionada actualmente para mostrar valores reales en vivo
  let liveObj = null;
  let liveLabel = "";
  if (ent === "nodo") {{
    if (agySeleccion && agySeleccion.tipo === "nodo") {{
      liveObj = AGY_DATA.nodos.find(x => x.id === agySeleccion.id) || null;
    }}
    if (liveObj) liveLabel = `Vértice activo: ${{liveObj.id}} (${{liveObj.nombre}})`;
  }} else {{
    if (agySeleccion && agySeleccion.tipo === "arista") {{
      liveObj = AGY_DATA.aristas.find(x => x.id === agySeleccion.id) || null;
    }}
    if (liveObj) liveLabel = `Arista activa: ${{liveObj.origen}} → ${{liveObj.destino}}`;
  }}

  const switchHtml = `
    <div class="agy-vm-switch">
      <button type="button" class="agy-vm-entity-btn ${{ent === 'nodo' ? 'active' : ''}}" onclick="agyAbrirModalVariables('nodo', 1)">Nodos · 12 Variables (nodo.py)</button>
      <button type="button" class="agy-vm-entity-btn ${{ent === 'arista' ? 'active' : ''}}" onclick="agyAbrirModalVariables('arista', 0)">Aristas · 20 Variables + Fórmulas (arista.py)</button>
    </div>
  `;

  const tabsHtml = `
    <div class="agy-vm-tabs">
      ${{cfg.secciones.map(s => `
        <button type="button" class="agy-vm-tab ${{s.id === sec.id ? 'active' : ''}}" onclick="agyAbrirModalVariables('${{ent}}', ${{s.id}})">${{s.label}}</button>
      `).join("")}}
    </div>
  `;

  const cardsHtml = sec.vars.map(v => {{
    const isCalc = v.origen.toLowerCase().includes("calculado") || v.origen.toLowerCase().includes("derivado");
    const liveVal = (liveObj && typeof v.live === "function") ? v.live(liveObj) : null;
    return `
      <div class="agy-vm-card">
        <div class="agy-vm-card-top">
          <span class="agy-vm-name">${{v.name}}</span>
          <span class="agy-vm-badges">
            <span class="agy-vm-tag">${{v.type}}</span>
            <span class="agy-vm-tag ${{isCalc ? 'calc' : ''}}">${{v.origen}}</span>
          </span>
        </div>
        <p class="agy-vm-desc">${{v.desc}}</p>
        ${{v.formula ? `
          <div class="agy-vm-formula">
            <div class="agy-vm-formula-cap">${{v.formulaCap || "Fórmula aplicada"}}</div>
            ${{v.formula}}
          </div>
        ` : ""}}
        <div class="agy-vm-meta">
          <span>Uso en código: <code>${{v.where}}</code></span>
          ${{liveVal !== null ? `<span class="agy-vm-live">Valor actual: <strong>${{liveVal}}</strong></span>` : ""}}
        </div>
      </div>
    `;
  }}).join("");

  content.innerHTML = `
    <span class="agy-badge" style="color:var(--color-warm-cream);margin-bottom:8px;">DICCIONARIO TÉCNICO Y FÓRMULAS DEL GRAFO${{liveLabel ? " · " + liveLabel : ""}}</span>
    <h2 id="agy-modal-title" style="margin:0 0 14px 0;font-size:22px;line-height:1.25;">${{cfg.titulo}}</h2>
    ${{switchHtml}}
    ${{tabsHtml}}
    <p class="agy-vm-lead">${{sec.intro}}</p>
    <div class="agy-vm-list">${{cardsHtml}}</div>
  `;
  backdrop.classList.add("open");
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
      <span class="agy-badge" style="color:var(--color-saffron-spark);"><span class="agy-tex"><var>w</var><sub><i>ij</i></sub> <span class="agy-op">=</span> ${{a.peso_combinado.toFixed(4)}}</span></span>
    </div>
    <p style="font-size:14px;">Distancia: ${{a.distancia_km}} km · Tiempo efectivo: ${{a.tiempo_efectivo_h.toFixed(2)}} h · Costo total: $${{a.costo_total_cop.toLocaleString()}} COP · Estado: ${{a.estado_via}}</p>
    <div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span><span class="agy-tex"><var>&alpha;</var> <span class="agy-op">=</span> 0.25</span> · Tiempo efectivo (${{a.tiempo_efectivo_h.toFixed(2)}} h)</span><span>+${{d.t_term.toFixed(4)}} (${{pct(d.t_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.t_term)}}%"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span><span class="agy-tex"><var>&beta;</var> <span class="agy-op">=</span> 0.20</span> · Costo total ($${{a.costo_total_cop.toLocaleString()}} COP)</span><span>+${{d.c_term.toFixed(4)}} (${{pct(d.c_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.c_term)}}%;background:#2997ff;"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span><span class="agy-tex"><var>&gamma;</var> <span class="agy-op">=</span> 0.20</span> · Riesgo combinado (<span class="agy-tex"><var>r</var><sub>comb</sub> <span class="agy-op">=</span> ${{a.riesgo_combinado.toFixed(3)}}</span>)</span><span>+${{d.r_term.toFixed(4)}} (${{pct(d.r_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.r_term)}}%;background:#ffb829;"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span><span class="agy-tex"><var>&delta;</var> <span class="agy-op">=</span> 0.10</span> · Pérdida térmica (${{a.perdida_esperada.toFixed(2)}}%)</span><span>+${{d.p_term.toFixed(4)}} (${{pct(d.p_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.p_term)}}%;background:#dc5000;"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span><span class="agy-tex"><var>&epsilon;</var> <span class="agy-op">=</span> 0.10</span> · Incertidumbre vial (<span class="agy-tex">1 <span class="agy-op">&minus;</span> <var>q</var><sub><i>ij</i></sub> <span class="agy-op">=</span> 1 <span class="agy-op">&minus;</span> ${{a.confiabilidad_via.toFixed(2)}}</span>)</span><span>+${{d.conf_term.toFixed(4)}} (${{pct(d.conf_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.conf_term)}}%;background:#15846e;"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span><span class="agy-tex"><var>&zeta;</var> <span class="agy-op">=</span> 0.05</span> · Probabilidad de retraso (<span class="agy-tex"><var>P</var><sub>ret</sub> <span class="agy-op">=</span> ${{a.prob_retraso.toFixed(2)}}</span>)</span><span>+${{d.ret_term.toFixed(4)}} (${{pct(d.ret_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.ret_term)}}%;background:#ec4899;"></div></div>
      </div>
    </div>
  `;
}}

let agyModoDestinoFLP = "PG";
let agyCandidatoFocusFLP = "F02";

const AGY_FLP_TIP_CATALOG = {{
  // ── 3.1 VARIABLES DE LOS NODOS (nodo.py) ──
  "node_id": {{
    cat: "VARIABLE DE NODO · nodo.py",
    title: "id",
    fuente: "nodos_pesca.csv · nodos_centros.csv · nodos_destinos.csv",
    tipo: "str",
    rol: "Clave primaria en GrafoLogistico (Tabla Hash O(1)). Identifica orígenes P01..P07, candidatos F01..F03, A01..A02 y destinos V_PG1/V_BOG1.",
    formula: "obtener_nodo(id) → O(1) en dict._nodos"
  }},
  "node_nombre": {{
    cat: "VARIABLE DE NODO · nodo.py",
    title: "nombre",
    fuente: "Los 3 archivos CSV de nodos",
    tipo: "str",
    rol: "Etiqueta descriptiva real de la asociación pesquera, centro frigorífico, acopio, muelle o mercado mayorista.",
    formula: "Metadato legible en informes, mapas y tablas"
  }},
  "node_tipo": {{
    cat: "VARIABLE DE NODO · nodo.py",
    title: "tipo",
    fuente: "Los 3 archivos CSV (P, A, F, M, D, H, V)",
    tipo: "str",
    rol: "Determina candidatos FLP (tipo ∈ {{'F','A'}}), paradas con abatimiento térmico en cold_chain.py, transbordos en M (+0.25 h) y multiplicador ×1.5 en D.",
    formula: "es_candidato_frio: tipo in ('F', 'A')"
  }},
  "node_coords": {{
    cat: "VARIABLE DE NODO · nodo.py",
    title: "latitud, longitud",
    fuente: "Los 3 CSV (origen: constants.py, manual commit 9af49ef, puntos_campo.geojson)",
    tipo: "float (WGS84)",
    rol: "Ubicación geográfica para cálculo Haversine, trazado OSRM/ríos y posicionamiento en Mapa SIG.",
    formula: "d = 2·R·arcsin(√(sin²(Δφ/2) + cos φ₁·cos φ₂·sin²(Δλ/2)))"
  }},
  "node_produccion": {{
    cat: "VARIABLE DE NODO · nodo.py",
    title: "produccion_kg_dia",
    fuente: "nodos_pesca.csv",
    tipo: "float (kg/día)",
    rol: "Oferta diaria pescada por asociación (265.75 kg/día totales). Entra en la restricción de capacidad del FLP (MILP/Greedy) y como lote inicial en SimuladorCadenaFrio.",
    formula: "Σ_i (produccion_i · x_ij) ≤ capacidad_frio_kg_j · y_j"
  }},
  "node_capacidad": {{
    cat: "VARIABLE DE NODO · nodo.py",
    title: "capacidad_frio_kg",
    fuente: "nodos_centros.csv",
    tipo: "float (kg)",
    rol: "Capacidad máxima de almacenamiento frigorífico (F01=2,000 · F02=5,000 · F03=4,000 · A01=1,500 · A02=1,200 kg). Además en cold_chain.py activa el reset térmico si > 0.",
    formula: "tiene_frio = (tipo ∈ {{'F','A'}}) ∧ (capacidad_frio_kg > 0)"
  }},
  "node_costo_fijo": {{
    cat: "VARIABLE DE NODO · nodo.py",
    title: "costo_fijo",
    fuente: "nodos_centros.csv",
    tipo: "float (COP/mes)",
    rol: "Costo fijo mensual f_j en facility_location.py (F01=$2.5M, F02=$3.0M, F03=$2.8M, A01=$1.2M, A02=$1.0M). NOTA: EvaluadorEscenarios usa la constante uniforme $5.0M COP/mes por planta.",
    formula: "min Z = Σ_j f_j·y_j + Σ_i Σ_j c_ij·x_ij"
  }},
  "node_energia": {{
    cat: "VARIABLE DE NODO · nodo.py",
    title: "tiene_energia",
    fuente: "nodos_centros.csv",
    tipo: "bool",
    rol: "Filtro duro de elegibilidad en Nodo.es_candidato_frio. Descarta automáticamente a A03 (Acopio Iteviare, tiene_energia=False).",
    formula: "es_candidato_frio requiere tiene_energia == True"
  }},
  "node_agua": {{
    cat: "VARIABLE DE NODO · nodo.py",
    title: "tiene_agua",
    fuente: "nodos_centros.csv",
    tipo: "bool",
    rol: "Atributo sanitario de infraestructura (disponibilidad de agua potable para fabricación de hielo en escarcha 1:1 y lavado).",
    formula: "Informativo en ficha de infraestructura (no bloquea FLP)"
  }},
  "node_muelle": {{
    cat: "VARIABLE DE NODO · nodo.py",
    title: "tiene_muelle",
    fuente: "nodos_centros.csv",
    tipo: "bool",
    rol: "Indica si el punto cuenta con embarcadero fluvial para descarga directa de canoas.",
    formula: "Informativo / coherencia multimodal (fluvial ↔ terrestre)"
  }},
  "node_restriccion": {{
    cat: "VARIABLE DE NODO · nodo.py",
    title: "restriccion_ambiental",
    fuente: "nodos_centros.csv",
    tipo: "bool",
    rol: "True en H01 (Humedal Manacacías) y H02 (Reserva Corocito). Bloquea el paso intermedio en Dijkstra, Bellman-Ford y Floyd-Warshall cuando evitar_restringidos=True, y excluye al nodo del FLP.",
    formula: "Si u.restriccion_ambiental → omitir u como nodo intermedio"
  }},
  "node_es_candidato": {{
    cat: "PROPIEDAD CALCULADA · nodo.py",
    title: "es_candidato_frio",
    fuente: "Propiedad @property en src/models/nodo.py",
    tipo: "bool (calculada)",
    rol: "Define el conjunto F de 5 candidatos válidos: F01, F02, A01, A02, F03 (descarta A03 por energía y H01/H02 por restricción ambiental).",
    formula: "tipo in ('F','A') and tiene_energia and not restriccion_ambiental"
  }},
  "node_metadata": {{
    cat: "VARIABLE DE NODO · nodo.py",
    title: "metadata (rio, pescadores, demanda_kg_dia)",
    fuente: "nodos_pesca.csv · nodos_destinos.csv",
    tipo: "dict",
    rol: "Almacena cuenca hidrográfica, número censado de pescadores (114 en 7 asociaciones) y demanda diaria de los mercados (V_PG1, V_BOG1).",
    formula: "Metadatos contextuales para reportes e itinerarios"
  }},
  // ── 3.2 VARIABLES DE LAS ARISTAS (arista.py) ──
  "edge_extremos": {{
    cat: "VARIABLE DE ARISTA · arista.py",
    title: "origen, destino",
    fuente: "aristas.csv (41 aristas dirigidas)",
    tipo: "str",
    rol: "Definen la topología dirigida E ⊆ V × V en la Lista de Adyacencia. Explican por qué F01 no recibe flujo del sur (∞) y por qué F03 no retorna a V_PG1.",
    formula: "_adyacencia[origen].append(arista)"
  }},
  "edge_distancia": {{
    cat: "VARIABLE DE ARISTA · arista.py",
    title: "distancia_km",
    fuente: "aristas.csv",
    tipo: "float (km)",
    rol: "Longitud física del tramo (ej. F02→F03 = 126.0 km por RN 40; P05→V_PG1 = 63.0 km totales). Usada como criterio alternativo en Dijkstra e itinerarios.",
    formula: "d_total = Σ distancia_km(e)"
  }},
  "edge_tiempo": {{
    cat: "VARIABLE DE ARISTA · arista.py",
    title: "tiempo_h",
    fuente: "aristas.csv",
    tipo: "float (horas)",
    rol: "CLAVE PRINCIPAL en la matriz 7×5 de facility_location.py (criterio='tiempo_h') y base del deterioro térmico en SimuladorCadenaFrio.",
    formula: "t_ruta = Σ tiempo_h(e) ≤ 4.0 h (límite FAO)"
  }},
  "edge_costo_base": {{
    cat: "VARIABLE DE ARISTA · arista.py",
    title: "costo_cop",
    fuente: "aristas.csv",
    tipo: "float (COP)",
    rol: "Flete base de transporte del tramo. Se suma con costo_peajes_cop para obtener costo_total_cop en MILP FLP y EvaluadorEscenarios.",
    formula: "costo_total_cop = costo_cop + costo_peajes_cop"
  }},
  "edge_peajes": {{
    cat: "VARIABLE DE ARISTA · arista.py",
    title: "peajes, costo_peajes_cop",
    fuente: "aristas.csv",
    tipo: "int, float (COP)",
    rol: "Número de casetas y valor de peajes viales (ej. F02→F03: 1 peaje = $3,000 COP). Integrado al costo total de transporte.",
    formula: "C_transporte(ruta) = Σ (costo_cop + costo_peajes_cop)"
  }},
  "edge_tipo_via": {{
    cat: "VARIABLE DE ARISTA · arista.py",
    title: "tipo_via",
    fuente: "aristas.csv (fluvial | acceso_local | terrestre)",
    tipo: "str",
    rol: "Separa las vistas Vial (Slide 3), Ríos (Slide 4) y Mixto (Slide 5), y define el estilo de trazado (rios.shp vs OSRM).",
    formula: "tipo_via ∈ {{'fluvial', 'acceso_local', 'terrestre'}}"
  }},
  "edge_estado_via": {{
    cat: "VARIABLE DE ARISTA · arista.py",
    title: "estado_via (factor_estado_via)",
    fuente: "aristas.csv (buena=1.0 · regular=1.2 · mala=1.5 · critica=2.0)",
    tipo: "str → float",
    rol: "Penaliza tiempo_efectivo_h en el peso multicriterio según calidad física de carretera o trocha.",
    formula: "t_ef = tiempo_h · factor_estado_via · factor_flujo + tiempo_espera_h"
  }},
  "edge_flujo": {{
    cat: "VARIABLE DE ARISTA · arista.py",
    title: "flujo_vehicular (factor_flujo)",
    fuente: "aristas.csv (bajo=1.0 · medio=1.1 · alto=1.25)",
    tipo: "str → float",
    rol: "Multiplica el tiempo efectivo por congestión vehicular en corredores como la Ruta Nacional 40.",
    formula: "t_ef = tiempo_h · f_estado · f_flujo + t_espera"
  }},
  "edge_espera": {{
    cat: "VARIABLE DE ARISTA · arista.py",
    title: "tiempo_espera_h",
    fuente: "aristas.csv",
    tipo: "float (horas)",
    rol: "Tiempo de espera operativo en el tramo (ej. F02→F03 = 0.05 h), sumado directamente en tiempo_efectivo_h.",
    formula: "t_ef = tiempo_h · f_estado · f_flujo + tiempo_espera_h"
  }},
  "edge_tiempo_ef": {{
    cat: "PROPIEDAD CALCULADA · arista.py",
    title: "tiempo_efectivo_h",
    fuente: "Propiedad @property en src/models/arista.py",
    tipo: "float (horas)",
    rol: "Tiempo real ajustado por estado de vía, flujo y espera. Alimenta el factor α (0.25) del peso multicriterio y la pérdida esperada L_ij.",
    formula: "t_ef = tiempo_h · f_estado · f_flujo + tiempo_espera_h"
  }},
  "edge_costo_total": {{
    cat: "PROPIEDAD CALCULADA · arista.py",
    title: "costo_total_cop",
    fuente: "Propiedad @property en src/models/arista.py",
    tipo: "float (COP)",
    rol: "Suma flete base + peajes. Usada como costo de asignación c_ij en MILP FLP, costo de transporte en EvaluadorEscenarios y factor β (0.20) multicriterio.",
    formula: "C_tot = costo_cop + costo_peajes_cop"
  }},
  "edge_riesgos": {{
    cat: "VARIABLES DE ARISTA · arista.py",
    title: "riesgo_termico, riesgo_accidente, riesgo_derrumbe",
    fuente: "aristas.csv (escala 0.0 – 1.0)",
    tipo: "float [0, 1]",
    rol: "Componen riesgo_combinado con ponderación 50% térmico + 30% accidente + 20% derrumbe para el factor γ (0.20) del peso multicriterio.",
    formula: "r_comb = 0.50·r_term + 0.30·r_acc + 0.20·r_derr"
  }},
  "edge_perdida": {{
    cat: "PROPIEDAD CALCULADA · arista.py",
    title: "perdida_esperada (L_ij)",
    fuente: "Propiedad @property en src/models/arista.py",
    tipo: "float (%)",
    rol: "Porcentaje esperado de pérdida del lote en el tramo según tiempo efectivo y riesgo térmico. Alimenta el factor δ (0.10) del peso multicriterio.",
    formula: "p_esp = min(100, t_ef · 2.5 · (1 + 2.5 · r_term))"
  }},
  "edge_confiabilidad": {{
    cat: "VARIABLE DE ARISTA · arista.py",
    title: "confiabilidad_via (q_ij)",
    fuente: "aristas.csv (0.0 – 1.0)",
    tipo: "float [0, 1]",
    rol: "Probabilidad de operatividad continua del corredor. Su complemento (1 − q_ij) aporta el factor ε (0.10) en peso_combinado.",
    formula: "término_ε = 0.10 · (1 − confiabilidad_via)"
  }},
  "edge_retraso": {{
    cat: "VARIABLE DE ARISTA · arista.py",
    title: "prob_retraso (P_ret)",
    fuente: "aristas.csv (0.0 – 1.0)",
    tipo: "float [0, 1]",
    rol: "Probabilidad de demora imprevista por clima/tráfico/cauce. Aporta el factor ζ (0.05) en peso_combinado.",
    formula: "término_ζ = 0.05 · prob_retraso"
  }},
  "edge_peso": {{
    cat: "PROPIEDAD CALCULADA · arista.py",
    title: "peso_combinado (w_ij)",
    fuente: "Propiedad @property en src/models/arista.py (ADR-02)",
    tipo: "float (adimensional)",
    rol: "Peso multicriterio de 6 factores usado por Dijkstra para seleccionar rutas en EvaluadorEscenarios e itinerarios.",
    formula: "w = 0.25·(t_ef/48) + 0.20·(C_tot/5M) + 0.20·r_comb + 0.10·(p_esp/100) + 0.10·(1−q) + 0.05·P_ret"
  }},
  "edge_capacidad": {{
    cat: "VARIABLE DE ARISTA · arista.py",
    title: "capacidad_kg, restriccion_peso_ton, horario_restriccion",
    fuente: "aristas.csv",
    tipo: "float, float, str",
    rol: "Capacidad física del vehículo/lancha (validada para masa bruta 2× con hielo 1:1 = 531.5 kg/día) y restricciones viales.",
    formula: "carga_bruta(pescado + hielo 1:1) ≤ capacidad_kg"
  }},
  // ── 3.3 CONSTANTES DEL MODELO (constants.py) ──
  "const_tmax": {{
    cat: "CONSTANTE · constants.py",
    title: "TIEMPO_MAX_SIN_FRIO_H = 4.0 h",
    fuente: "src/utils/constants.py (Codex Alimentarius FAO/OMS)",
    tipo: "float (4.0 horas = 240 min)",
    rol: "Umbral máximo de cobertura sanitaria en facility_location.py, límite de poda en dijkstra_con_cota_inocuidad y criterio cumple_limite en cold_chain.py.",
    formula: "t_sin_frío ≤ 4.0 h (previene formación irreversible de histamina)"
  }},
  "const_temp": {{
    cat: "CONSTANTE · constants.py",
    title: "TEMP_AMBIENTE_LLANOS_C = 30.0 °C / TEMP_OBJETIVO_C = 0.0 °C",
    fuente: "src/utils/constants.py",
    tipo: "float (°C)",
    rol: "Temperatura ambiente de Puerto Gaitán (30 °C → tasa base 2.5%/h) vs. temperatura de conservación con hielo 1:1 (0 °C → vida útil 360 h).",
    formula: "T_amb = 30.0 °C → r(30°C) = 2.5%/h · T_obj = 0.0 °C"
  }},
  "const_deterioro": {{
    cat: "CONSTANTE · constants.py",
    title: "FACTOR_DETERIORO_ZONA_D = 1.5",
    fuente: "src/utils/constants.py (ADR-04)",
    tipo: "float (×1.5)",
    rol: "Multiplica el deterioro del tramo entrante al cruzar una zona crítica D01 (Iteviare) o D02 (La Culebra): pasa de 2.5%/h a 3.75%/h.",
    formula: "Δd_D = calcular_deterioro(t, 30°C) × 1.5"
  }},
  "const_muelle": {{
    cat: "CONSTANTE · constants.py",
    title: "TIEMPO_TRANSBORDO_MUELLE_H = 0.25 h",
    fuente: "src/utils/constants.py (ADR-04)",
    tipo: "float (0.25 h = 15 min)",
    rol: "Tiempo de espera sin frío añadido al cruzar un muelle M01/M02/M03 en SimuladorCadenaFrio (transbordo fluvial → terrestre).",
    formula: "t_tramo_M = tiempo_h + 0.25 h"
  }},
  "const_perdida_kg": {{
    cat: "CONSTANTE · constants.py",
    title: "COSTO_PERDIDA_KG = $12,000 COP/kg",
    fuente: "src/utils/constants.py",
    tipo: "int ($12,000 COP)",
    rol: "Valorización económica de cada kilogramo de pescado deteriorado (merma) en SimuladorCadenaFrio y EvaluadorEscenarios.",
    formula: "C_merma = kg_perdidos × $12,000 COP/kg"
  }},
  "const_fijo_esc": {{
    cat: "CONSTANTE · constants.py",
    title: "COSTO_FIJO_CENTRO_FRIO = $5,000,000 COP/mes",
    fuente: "src/utils/constants.py",
    tipo: "int ($5,000,000 COP/mes)",
    rol: "Costo fijo mensual uniforme que EvaluadorEscenarios multiplica por num_centros (1 planta = $5M, 3 plantas = $15M → ahorro de $10M COP/mes).",
    formula: "C_fijo_escenario = num_centros × $5,000,000 COP/mes"
  }},
  "const_pesos": {{
    cat: "CONSTANTES · constants.py",
    title: "α=0.25 · β=0.20 · γ=0.20 · δ=0.10 · ε=0.10 · ζ=0.05",
    fuente: "src/utils/constants.py (ADR-02)",
    tipo: "float (suman 0.90)",
    rol: "Ponderaciones de los 6 factores del peso multicriterio Arista.peso_combinado.",
    formula: "w = 0.25·t̂ + 0.20·ĉ + 0.20·r_comb + 0.10·p̂ + 0.10·(1−q) + 0.05·P_ret"
  }},
  // ── KPIs & PIPELINE ──
  "kpi_solo_f02": {{
    cat: "RESULTADO ÓPTIMO FLP",
    title: "Solo F02 (Centro Frío Puerto Gaitán)",
    fuente: "facility_location.py + scenario_evaluator.py",
    tipo: "1 planta abierta",
    rol: "Único candidato que alcanza cobertura 7/7 (100%) en la primera iteración Greedy y solución óptima del MILP PuLP.",
    formula: "Uso = 265.75 / 5,000 = 5.315% de capacidad frigorífica"
  }},
  "kpi_cobertura_fao": {{
    cat: "COBERTURA SANITARIA FAO",
    title: "7/7 asociaciones ≤ 4.0 h",
    fuente: "dijkstra.py (tiempo_h) & dijkstra_con_cota_inocuidad",
    tipo: "100.0%",
    rol: "En FLP (suma pura de aristas) el máximo es P05→F02 = 2.65 h. Con poda TCSPP que reinicia en A03/A01 el tramo máximo sin frío es 2.10 h.",
    formula: "max_i d_tiempo(i, F02) = 2.65 h ≤ 4.00 h (margen +1.35 h)"
  }},
  "kpi_costo_opt": {{
    cat: "COSTO TOTAL MENSUAL",
    title: "$13.02M Bogotá · $5.52M Local",
    fuente: "scenario_evaluator.py (EvaluadorEscenarios)",
    tipo: "COP / mes",
    rol: "Suma costo fijo uniforme ($5,000,000) + transporte mensual (×30 días) + costo de merma térmica ($12,000/kg).",
    formula: "C_total = $5,000,000 + C_transporte + C_merma"
  }},
  "kpi_ahorro_flp": {{
    cat: "AHORRO FRENTE A 3 PLANTAS",
    title: "−$10,000,000 COP/mes (−$120M/año)",
    fuente: "Comparación Escenario 3 (Solo F02) vs Escenario 5 (F01+F02+F03)",
    tipo: "−43.4% Bogotá / −64.4% Local",
    rol: "Evita abrir F01 y F03 simultáneamente ($15M fijo vs $5M fijo) manteniendo exactamente la misma cobertura 100% y la misma merma.",
    formula: "ΔC = $23,023,272 − $13,023,272 = $10,000,000 COP/mes"
  }},
  "pipe_csv": {{
    cat: "PASO 1 DEL PIPELINE · ENTRADA",
    title: "data/csv/nodos_centros.csv (13 nodos)",
    fuente: "src/utils/data_loader.py → cargar_grafo()",
    tipo: "CSV tabular",
    rol: "Contiene 3 Frío (F01–F03), 3 Acopio (A01–A03), 3 Muelles (M01–M03), 2 Deterioro (D01–D02) y 2 Hábitats (H01–H02).",
    formula: "Total grafo G=(V,E): 7 P + 13 Centros + 5 V = 25 vértices"
  }},
  "pipe_filtro": {{
    cat: "PASO 2 DEL PIPELINE · FILTRO BOOLEANO",
    title: "GrafoLogistico.candidatos_frio()",
    fuente: "src/models/nodo.py + src/models/grafo_logistico.py",
    tipo: "Filtro O(|V|)",
    rol: "Evalúa la propiedad booleana es_candidato_frio en cada vértice antes de ejecutar la matriz de Dijkstra.",
    formula: "tipo ∈ {{'F','A'}} ∧ tiene_energia=True ∧ ¬restriccion_ambiental"
  }},
  "pipe_salida": {{
    cat: "PASO 3 DEL PIPELINE · CONJUNTO F",
    title: "5 Candidatos Válidos + A03 Descartado",
    fuente: "Salida de grafo.candidatos_frio()",
    tipo: "List[Nodo] (len = 5)",
    rol: "Pasan F01, F02, A01, A02, F03. Se rechaza A03 (Acopio Iteviare) por tiene_energia=False y H01/H02 por restriccion_ambiental=True.",
    formula: "F = {{F01, F02, A01, A02, F03}} → Matriz Dijkstra 7 × 5 = 35 corridas"
  }}
}};

function agyHtmlTipDesdeObj(info) {{
  if (!info) return "";
  const filas = [];
  if (info.fuente) filas.push(`<div class="agy-ht-row"><span class="agy-ht-k">Archivo fuente / origen</span><span class="agy-ht-v"><code>${{info.fuente}}</code></span></div>`);
  if (info.tipo) filas.push(`<div class="agy-ht-row"><span class="agy-ht-k">Tipo / Valor</span><span class="agy-ht-v">${{info.tipo}}</span></div>`);
  if (info.rol) filas.push(`<div class="agy-ht-row"><span class="agy-ht-k">Rol exacto en el cálculo</span><span class="agy-ht-v">${{info.rol}}</span></div>`);
  if (info.formula) filas.push(`<div class="agy-ht-row"><span class="agy-ht-k">Fórmula / Regla aplicada</span><span class="agy-ht-v"><code>${{info.formula}}</code></span></div>`);
  return `
    <span class="agy-ht-cat">${{info.cat || "DETALLE TÉCNICO"}}</span>
    <div class="agy-ht-title">${{info.title || ""}}</div>
    <div class="agy-ht-grid">${{filas.join("")}}</div>
  `;
}}

function agyResolverContenidoHoverTip(el) {{
  const key = el.getAttribute("data-flp-tip");
  if (key && AGY_FLP_TIP_CATALOG[key]) {{
    return agyHtmlTipDesdeObj(AGY_FLP_TIP_CATALOG[key]);
  }}
  const cellAttr = el.getAttribute("data-flp-cell");
  if (cellAttr) {{
    const [origId, candId] = cellAttr.split("|");
    const det = AGY_DATA.flp_detallado || {{}};
    const fila = (det.matriz_7x5 || []).find(r => r.origen_id === origId);
    const c = fila && fila.celdas ? fila.celdas[candId] : null;
    if (c) {{
      const tramosStr = (c.tramos || []).map(t => `${{t.u}}→${{t.v}} (${{t.tiempo_h.toFixed(2)}} h)`).join(" + ");
      return agyHtmlTipDesdeObj({{
        cat: `CELDA DIJKSTRA 7×5 · ${{origId}} → ${{candId}}`,
        title: c.encontrado ? `${{c.tiempo_h.toFixed(2)}} h (${{c.cumple_fao ? "✓ Cumple FAO ≤ 4.0 h" : "⚠ Viola FAO > 4.0 h"}})` : "∞ (No alcanzable)",
        fuente: "src/algorithms/dijkstra.py (criterio='tiempo_h', evitar_restringidos=True)",
        tipo: c.encontrado ? `Distancia: ${{c.dist_km}} km · Costo flete+peajes: $${{(c.costo_cop || 0).toLocaleString()}} COP` : "Sin camino dirigido en G=(V,E)",
        rol: c.encontrado ? `Ruta exacta: ${{c.camino.join(" → ")}}${{c.nota ? " · " + c.nota : ""}}` : (c.nota || "No existe trayectoria dirigida."),
        formula: c.encontrado ? (tramosStr + ` = ${{c.tiempo_h.toFixed(2)}} h`) : "d_tiempo(u, v) = +∞"
      }});
    }}
  }}
  const stepAttr = el.getAttribute("data-flp-step");
  if (stepAttr !== null && stepAttr !== "") {{
    const idx = parseInt(stepAttr, 10);
    const pasos = (AGY_DATA.flp_detallado && AGY_DATA.flp_detallado.simulacion_p05 && AGY_DATA.flp_detallado.simulacion_p05.pasos) || [];
    const p = pasos[idx];
    if (p) {{
      return agyHtmlTipDesdeObj({{
        cat: `PASO ${{p.paso}} · SIMULACIÓN TÉRMICA P05 (cold_chain.py)`,
        title: `${{p.nodo_id}} · ${{p.nombre}} (Tipo ${{p.tipo}})`,
        fuente: "SimuladorCadenaFrio.simular_ruta(ruta_p05, 30.0 kg)",
        tipo: `Temp: ${{p.temp_c}} °C · Carga viva: ${{p.kg_actual.toFixed(3)}} kg (Δ = −${{p.kg_perdidos_paso.toFixed(3)}} kg)`,
        rol: `Efecto: ${{p.efecto}} · Tiempo acum.: ${{p.tiempo_acum_h.toFixed(2)}} h · Timer sin frío: ${{p.tiempo_sin_frio_h.toFixed(2)}} h · Det. acum.: ${{p.det_acum_pct.toFixed(3)}}%`,
        formula: p.regla_matematica
      }});
    }}
  }}
  const escAttr = el.getAttribute("data-flp-esc");
  if (escAttr) {{
    const [modo, idxStr] = escAttr.split("|");
    const idx = parseInt(idxStr, 10);
    const lista = modo === "PG" ? (AGY_DATA.escenarios_pg || []) : (AGY_DATA.escenarios_bog || []);
    const e = lista[idx];
    if (e) {{
      const fijoM = (e.num_centros * 5).toFixed(0);
      return agyHtmlTipDesdeObj({{
        cat: `ESCENARIO ${{idx + 1}} · ${{modo === "PG" ? "LOCAL (V_PG1)" : "BOGOTÁ (V_BOG1)"}}`,
        title: e.nombre,
        fuente: "src/optimization/scenario_evaluator.py → EvaluadorEscenarios",
        tipo: `Total: $${{Math.round(e.costo_total_cop).toLocaleString()}} COP/mes · Unitario: $${{Math.round(e.costo_por_kg_cop).toLocaleString()}} COP/kg`,
        rol: `Centros: [${{(e.centros_ids || []).join(", ") || "Ninguno"}}] · Cobertura térmica: ${{e.cumplimiento_termico_pct.toFixed(1)}}% · Merma: ${{e.pct_perdida.toFixed(2)}}% (${{e.kg_perdido.toFixed(1)}} kg/día)`,
        formula: `C_total = (${{e.num_centros}} × $5M = $${{fijoM}}M fijo) + C_transporte + C_merma`
      }});
    }}
  }}
  const provAttr = el.getAttribute("data-flp-prov");
  if (provAttr) {{
    const cands = (AGY_DATA.flp_detallado && AGY_DATA.flp_detallado.candidatos) || [];
    const c = cands.find(x => x.id === provAttr);
    if (c) {{
      return agyHtmlTipDesdeObj({{
        cat: `ORIGEN DE COORDENADAS · ${{c.origen_commit}}`,
        title: `${{c.id}} · ${{c.nombre}} (${{c.lat}}, ${{c.lon}})`,
        fuente: `data/csv/nodos_centros.csv (${{c.origen_commit}})`,
        tipo: `Tipo ${{c.tipo}} · Cap: ${{c.capacidad_frio_kg.toLocaleString()}} kg · Fijo CSV: $${{c.costo_fijo_csv.toLocaleString()}} COP`,
        rol: `${{c.origen_coords}}. ${{c.regla}}`,
        formula: c.es_candidato ? `Candidato válido → Cobertura ${{c.cobertura_n}}/7 (${{c.cobertura_pct.toFixed(1)}}%)` : "DESCARTADO: tiene_energia = False"
      }});
    }}
  }}
  return "";
}}

function agyPosicionarHoverTip(ev) {{
  const tip = document.getElementById("agy-flp-hover-tip");
  if (!tip || !tip.classList.contains("visible")) return;
  const pad = 14;
  const tw = tip.offsetWidth || 370;
  const th = tip.offsetHeight || 170;
  let x = ev.clientX + 16;
  let y = ev.clientY + 16;
  if (x + tw + pad > window.innerWidth) x = Math.max(pad, ev.clientX - tw - 16);
  if (y + th + pad > window.innerHeight) y = Math.max(pad, ev.clientY - th - 16);
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}}

let agyHoverTooltipsInit = false;
function agyInicializarHoverTooltipsFLP() {{
  if (agyHoverTooltipsInit) return;
  agyHoverTooltipsInit = true;
  const tip = document.getElementById("agy-flp-hover-tip");
  if (!tip) return;
  let activeEl = null;

  document.addEventListener("mouseover", (ev) => {{
    const target = ev.target && ev.target.closest
      ? ev.target.closest("[data-flp-tip],[data-flp-cell],[data-flp-step],[data-flp-esc],[data-flp-prov]")
      : null;
    if (!target) return;
    if (target !== activeEl) {{
      const html = agyResolverContenidoHoverTip(target);
      if (!html) return;
      activeEl = target;
      tip.innerHTML = html;
      tip.classList.add("visible");
      tip.setAttribute("aria-hidden", "false");
    }}
    agyPosicionarHoverTip(ev);
  }});

  document.addEventListener("mousemove", (ev) => {{
    if (!activeEl) return;
    const stillInside = ev.target && ev.target.closest
      ? ev.target.closest("[data-flp-tip],[data-flp-cell],[data-flp-step],[data-flp-esc],[data-flp-prov]")
      : null;
    if (!stillInside) {{
      activeEl = null;
      tip.classList.remove("visible");
      tip.setAttribute("aria-hidden", "true");
      return;
    }}
    agyPosicionarHoverTip(ev);
  }});

  document.addEventListener("mouseout", (ev) => {{
    if (!activeEl) return;
    const rel = ev.relatedTarget && ev.relatedTarget.closest
      ? ev.relatedTarget.closest("[data-flp-tip],[data-flp-cell],[data-flp-step],[data-flp-esc],[data-flp-prov]")
      : null;
    if (!rel) {{
      activeEl = null;
      tip.classList.remove("visible");
      tip.setAttribute("aria-hidden", "true");
    }}
  }});
}}

function agyCambiarSubtabFLP(idx) {{
  for (let i = 0; i < 4; i++) {{
    document.getElementById(`agy-flp-tab-${{i}}`)?.classList.toggle("active", i === idx);
    document.getElementById(`agy-flp-pane-${{i}}`)?.classList.toggle("active", i === idx);
  }}
}}

function agySeleccionarCandidatoFLP(candId) {{
  agyCandidatoFocusFLP = candId || "F02";
  agyRenderizarMatriz7x5FLP();
  agyCambiarSubtabFLP(1);
}}

function agyRenderizarWorkspaceFLP() {{
  const det = AGY_DATA.flp_detallado || {{}};
  const cands = det.candidatos || [];

  // ── SUBTAB 0: ORIGEN, FILTRO Y CATÁLOGO INTERACTIVO DE VARIABLES ──
  const originWrap = document.getElementById("agy-flp-origin-wrap");
  if (originWrap) {{
    const filasProv = cands.map(c => `
      <tr class="esc-hoverable ${{c.id === 'F02' ? 'esc-opt' : ''}}" data-flp-prov="${{c.id}}">
        <td style="font-family:var(--font-mono);font-weight:700;">${{c.id === 'F02' ? '★ ' : ''}}${{c.id}}</td>
        <td>${{c.nombre}}</td>
        <td style="font-family:var(--font-mono);">${{c.lat.toFixed(4)}}, ${{c.lon.toFixed(4)}}</td>
        <td>${{c.origen_coords}}</td>
        <td style="text-align:right;font-family:var(--font-mono);">${{c.capacidad_frio_kg.toLocaleString()}} kg</td>
        <td style="text-align:right;font-family:var(--font-mono);">$${{c.costo_fijo_csv.toLocaleString()}}</td>
        <td>${{c.es_candidato ? '<span style="color:#34d399;font-weight:700;">✓ Candidato</span>' : '<span style="color:#f87171;font-weight:700;">✗ Sin energía</span>'}}</td>
      </tr>
    `).join("");

    originWrap.innerHTML = `
      <div class="agy-flp-pipe">
        <div class="agy-flp-pipe-box agy-tip-target" data-flp-tip="pipe_csv">
          <div class="agy-flp-pipe-kicker">1. Entrada CSV (nodos_centros.csv)</div>
          <div class="agy-flp-pipe-title">${{det.num_centros_csv || 13}} Nodos Logísticos</div>
          <div style="font-size:12.5px;color:#a1a1aa;">3 Frío (F01–F03) · 3 Acopio (A01–A03) · 3 Muelles (M01–M03) · 2 Deterioro (D01–D02) · 2 Hábitats (H01–H02)</div>
        </div>
        <div class="agy-flp-pipe-arrow">→</div>
        <div class="agy-flp-pipe-box accent agy-tip-target" data-flp-tip="pipe_filtro">
          <div class="agy-flp-pipe-kicker">2. Filtro Booleano (nodo.py)</div>
          <div class="agy-flp-pipe-title">Nodo.es_candidato_frio</div>
          <div style="font-family:var(--font-mono);font-size:11.5px;color:#c4b5fd;">tipo ∈ {{'F','A'}} ∧ tiene_energia ∧ ¬restriccion_ambiental</div>
        </div>
        <div class="agy-flp-pipe-arrow">→</div>
        <div class="agy-flp-pipe-box agy-tip-target" data-flp-tip="pipe_salida">
          <div class="agy-flp-pipe-kicker">3. Salida del Filtro (Conjunto F)</div>
          <div class="agy-flp-pipe-title">5 Candidatos Válidos Evaluados</div>
          <div class="agy-flp-var-chips" style="margin-top:6px;">
            <span class="agy-flp-vchip" data-flp-prov="F01" onclick="agySeleccionarCandidatoFLP('F01')">F01</span>
            <span class="agy-flp-vchip opt" data-flp-prov="F02" onclick="agySeleccionarCandidatoFLP('F02')">★ F02</span>
            <span class="agy-flp-vchip" data-flp-prov="A01" onclick="agySeleccionarCandidatoFLP('A01')">A01</span>
            <span class="agy-flp-vchip" data-flp-prov="A02" onclick="agySeleccionarCandidatoFLP('A02')">A02</span>
            <span class="agy-flp-vchip" data-flp-prov="F03" onclick="agySeleccionarCandidatoFLP('F03')">F03</span>
            <span class="agy-flp-vchip rejected" data-flp-prov="A03">✗ A03 (sin energía)</span>
          </div>
        </div>
      </div>

      <div class="agy-flp-var-grid">
        <div class="agy-flp-var-col">
          <div class="agy-flp-var-col-head"><span>3.1 VARIABLES DE NODOS (nodo.py)</span><span style="color:#a1a1aa;">Hover para ficha</span></div>
          <div class="agy-flp-var-chips">
            <span class="agy-flp-vchip" data-flp-tip="node_id">id</span>
            <span class="agy-flp-vchip" data-flp-tip="node_nombre">nombre</span>
            <span class="agy-flp-vchip" data-flp-tip="node_tipo">tipo</span>
            <span class="agy-flp-vchip" data-flp-tip="node_coords">latitud, longitud</span>
            <span class="agy-flp-vchip" data-flp-tip="node_produccion">produccion_kg_dia</span>
            <span class="agy-flp-vchip" data-flp-tip="node_capacidad">capacidad_frio_kg</span>
            <span class="agy-flp-vchip" data-flp-tip="node_costo_fijo">costo_fijo</span>
            <span class="agy-flp-vchip" data-flp-tip="node_energia">tiene_energia</span>
            <span class="agy-flp-vchip" data-flp-tip="node_agua">tiene_agua</span>
            <span class="agy-flp-vchip" data-flp-tip="node_muelle">tiene_muelle</span>
            <span class="agy-flp-vchip" data-flp-tip="node_restriccion">restriccion_ambiental</span>
            <span class="agy-flp-vchip calc" data-flp-tip="node_es_candidato">es_candidato_frio ★</span>
            <span class="agy-flp-vchip" data-flp-tip="node_metadata">metadata</span>
          </div>
        </div>

        <div class="agy-flp-var-col">
          <div class="agy-flp-var-col-head"><span>3.2 VARIABLES DE ARISTAS (arista.py)</span><span style="color:#a1a1aa;">Hover para ficha</span></div>
          <div class="agy-flp-var-chips">
            <span class="agy-flp-vchip" data-flp-tip="edge_extremos">origen, destino</span>
            <span class="agy-flp-vchip" data-flp-tip="edge_distancia">distancia_km</span>
            <span class="agy-flp-vchip opt" data-flp-tip="edge_tiempo">tiempo_h ★</span>
            <span class="agy-flp-vchip" data-flp-tip="edge_costo_base">costo_cop</span>
            <span class="agy-flp-vchip" data-flp-tip="edge_peajes">peajes, costo_peajes_cop</span>
            <span class="agy-flp-vchip" data-flp-tip="edge_tipo_via">tipo_via</span>
            <span class="agy-flp-vchip" data-flp-tip="edge_estado_via">estado_via</span>
            <span class="agy-flp-vchip" data-flp-tip="edge_flujo">flujo_vehicular</span>
            <span class="agy-flp-vchip" data-flp-tip="edge_espera">tiempo_espera_h</span>
            <span class="agy-flp-vchip" data-flp-tip="edge_riesgos">riesgo_term/acc/derr</span>
            <span class="agy-flp-vchip" data-flp-tip="edge_confiabilidad">confiabilidad_via</span>
            <span class="agy-flp-vchip" data-flp-tip="edge_retraso">prob_retraso</span>
            <span class="agy-flp-vchip calc" data-flp-tip="edge_tiempo_ef">tiempo_efectivo_h</span>
            <span class="agy-flp-vchip calc" data-flp-tip="edge_costo_total">costo_total_cop</span>
            <span class="agy-flp-vchip calc" data-flp-tip="edge_perdida">perdida_esperada</span>
            <span class="agy-flp-vchip calc" data-flp-tip="edge_peso">peso_combinado</span>
            <span class="agy-flp-vchip" data-flp-tip="edge_capacidad">capacidad_kg</span>
          </div>
        </div>

        <div class="agy-flp-var-col">
          <div class="agy-flp-var-col-head"><span>3.3 CONSTANTES (constants.py)</span><span style="color:#a1a1aa;">Hover para ficha</span></div>
          <div class="agy-flp-var-chips">
            <span class="agy-flp-vchip const" data-flp-tip="const_tmax">TIEMPO_MAX_SIN_FRIO_H = 4.0 h</span>
            <span class="agy-flp-vchip const" data-flp-tip="const_temp">T_amb = 30 °C · T_obj = 0 °C</span>
            <span class="agy-flp-vchip const" data-flp-tip="const_deterioro">FACTOR_DETERIORO_ZONA_D = ×1.5</span>
            <span class="agy-flp-vchip const" data-flp-tip="const_muelle">TIEMPO_TRANSBORDO_MUELLE = 0.25 h</span>
            <span class="agy-flp-vchip const" data-flp-tip="const_perdida_kg">COSTO_PERDIDA_KG = $12,000/kg</span>
            <span class="agy-flp-vchip const" data-flp-tip="const_fijo_esc">COSTO_FIJO_CENTRO_FRIO = $5M/mes</span>
            <span class="agy-flp-vchip const" data-flp-tip="const_pesos">Pesos α..ζ (0.25..0.05)</span>
          </div>
        </div>
      </div>

      <div class="agy-plinth-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
          <div>
            <h3 style="margin:0;">Origen Verificado de Coordenadas de los Candidatos (Git &amp; CSV)</h3>
            <p style="font-size:13px;margin:4px 0 0 0;">Pasa el cursor sobre cualquier fila para ver la trazabilidad exacta del commit, regla geográfica y estado en el FLP.</p>
          </div>
          <span class="agy-badge" style="color:#c4b5fd;border-color:rgba(128,82,255,0.4);">commit 9af49ef + aaab5c5</span>
        </div>
        <div style="overflow-x:auto;">
          <table class="agy-flp-esc-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Nombre del Nodo</th>
                <th>Coordenadas WGS84</th>
                <th>Origen de las Coordenadas</th>
                <th style="text-align:right;">Capacidad</th>
                <th style="text-align:right;">Fijo CSV</th>
                <th>Estado FLP</th>
              </tr>
            </thead>
            <tbody>${{filasProv}}</tbody>
          </table>
        </div>
        <div style="margin-top:10px;font-size:12.5px;color:#a1a1aa;">
          <strong>Nota de campo:</strong> Los 7 nodos de infraestructura complementaria (<code>M01, M02, M03, D01, D02, H01, H02</code>) se incorporaron posteriormente en el <code>commit aaab5c5</code> desde el levantamiento real <code>data/geojson/puntos_campo.geojson</code>.
        </div>
      </div>
    `;
  }}

  agyRenderizarMatriz7x5FLP();
  agyRenderizarSimulacionP05yFormulasFLP();
  agyRenderizarEscenariosYAuditoriaFLP(agyModoDestinoFLP);
}}

function agyRenderizarMatriz7x5FLP() {{
  const matWrap = document.getElementById("agy-flp-matrix-wrap");
  const det = AGY_DATA.flp_detallado || {{}};
  if (!matWrap || !det.matriz_7x5) return;

  const cols = ["F01", "F02", "A01", "A02", "F03"];
  const candsMap = {{}};
  (det.candidatos || []).forEach(c => {{ candsMap[c.id] = c; }});

  const ths = cols.map(cid => {{
    const isOpt = cid === "F02";
    const isFoc = cid === agyCandidatoFocusFLP;
    const cls = (isOpt ? "col-opt " : "") + (isFoc ? "col-focus" : "");
    return `<th class="${{cls}}" onclick="agyCandidatoFocusFLP='${{cid}}';agyRenderizarMatriz7x5FLP();" title="Clic para resaltar ${{cid}}">${{isOpt ? "★ F02 (Óptimo)" : cid}}</th>`;
  }}).join("");

  const filas = det.matriz_7x5.map(row => {{
    const tds = cols.map(cid => {{
      const cell = row.celdas[cid];
      const isOpt = cid === "F02";
      if (!cell || !cell.encontrado) {{
        return `<td class="cell-hoverable cell-inf ${{isOpt ? 'col-opt' : ''}}" data-flp-cell="${{row.origen_id}}|${{cid}}">∞</td>`;
      }}
      let cls = "cell-hoverable ";
      let extra = "";
      if (isOpt) cls += "col-opt ";
      if (!cell.cumple_fao) {{
        cls += "cell-fail ";
        extra = " (>4h)";
      }} else if (row.origen_id === "P05" && cid === "F02") {{
        cls += "cell-warn ";
      }} else if (row.origen_id === "P04" && (cid === "A01" || cid === "F02")) {{
        extra = " 🌿";
      }}
      return `<td class="${{cls}}" data-flp-cell="${{row.origen_id}}|${{cid}}">${{cell.tiempo_h.toFixed(2)}} h${{extra}}</td>`;
    }}).join("");

    return `
      <tr>
        <td><strong>${{row.origen_id}}</strong> · ${{agyNombreCompacto(row.nombre)}} <span style="font-family:var(--font-mono);font-size:11px;color:#9ca3af;">(${{row.produccion_kg_dia.toFixed(1)}} kg/d)</span></td>
        ${{tds}}
      </tr>
    `;
  }}).join("");

  const footTds = cols.map(cid => {{
    const c = candsMap[cid] || {{ cobertura_n: 0, cobertura_pct: 0 }};
    const isOpt = cid === "F02";
    return `<td class="${{isOpt ? 'col-opt' : ''}}">${{c.cobertura_n}}/7 (${{c.cobertura_pct.toFixed(1)}}%)${{isOpt ? ' ★' : ''}}</td>`;
  }}).join("");

  // Tabla tramo a tramo hacia F02
  const filasF02 = det.matriz_7x5.map(row => {{
    const c = row.celdas["F02"];
    if (!c || !c.encontrado) return "";
    const sumStr = (c.tramos || []).map(t => `${{t.tiempo_h.toFixed(2)}}`).join(" + ");
    return `
      <tr class="esc-hoverable" data-flp-cell="${{row.origen_id}}|F02">
        <td style="font-family:var(--font-mono);font-weight:700;">${{row.origen_id}}</td>
        <td style="font-family:var(--font-mono);font-size:12px;">${{c.camino.join(" → ")}}</td>
        <td style="font-family:var(--font-mono);font-size:12px;color:#a1a1aa;">${{sumStr}}</td>
        <td style="text-align:right;font-family:var(--font-mono);font-weight:700;color:${{c.tiempo_h > 2.0 ? '#fbbf24' : '#34d399'}};">${{c.tiempo_h.toFixed(2)}} h</td>
        <td style="text-align:right;font-family:var(--font-mono);">${{c.dist_km.toFixed(1)}} km</td>
      </tr>
    `;
  }}).join("");

  matWrap.innerHTML = `
    <div class="agy-plinth-card" style="margin-bottom:16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px;">
        <div>
          <h3 style="margin:0;">Matriz de Tiempos Dijkstra 7×5 (criterio = <code>tiempo_h</code>, <code>evitar_restringidos = True</code>)</h3>
          <p style="font-size:13px;margin:4px 0 0 0;">Pasa el cursor sobre <strong style="color:#c4b5fd;">cualquier celda</strong> para ver la ruta exacta, la suma arista por arista, distancia y costo. El ícono 🌿 marca el desvío ecológico por <code>H01</code>.</p>
        </div>
        <span class="agy-delta-pill">★ Ganador Iteración 1: F02 (7/7 = 100%)</span>
      </div>
      <div style="overflow-x:auto;">
        <table class="agy-flp-mat-table">
          <thead>
            <tr>
              <th>Asociación Pesquera (Origen)</th>
              ${{ths}}
            </tr>
          </thead>
          <tbody>
            ${{filas}}
            <tr class="mat-foot">
              <td>Cobertura Sanitaria FAO (≤ 4.00 h)</td>
              ${{footTds}}
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="agy-bento-grid-2">
      <div class="agy-plinth-card">
        <h3>Algoritmo Greedy Constructivo vs. MILP PuLP (<code>facility_location.py</code>)</h3>
        <div class="agy-flp-formula-card" style="margin-top:10px;">
          <div class="agy-flp-formula-head">
            <span class="agy-flp-formula-title">Iteración 1 — Selección Voraz (Greedy)</span>
            <span class="agy-flp-formula-src">resolver_localizacion_greedy()</span>
          </div>
          <p style="font-size:13px;margin:0 0 8px 0;">Evalúa la ganancia de cobertura de cada candidato en ≤ 4.0 h: <code>F02=7</code>, <code>F03=6</code>, <code>F01=4</code>, <code>A02=4</code>, <code>A01=3</code>. Selecciona <strong>F02</strong> y alcanza <code>cubiertos = 7/7 (100%)</code> en una sola iteración → <strong>fin del algoritmo</strong>.</p>
          <div style="font-family:var(--font-mono);font-size:12px;color:#34d399;">centros_abiertos = ['F02'] · t_max = 2.65 h · costo_fijo_csv = $3,000,000 COP/mes</div>
        </div>
        <div class="agy-flp-formula-card" style="margin-bottom:0;">
          <div class="agy-flp-formula-head">
            <span class="agy-flp-formula-title">Verificación Exacta MILP (PuLP)</span>
            <span class="agy-flp-formula-src">resolver_localizacion_pulp()</span>
          </div>
          <div class="agy-vm-math-stack" style="margin:6px 0;">
            <div class="agy-vm-math-row">
              <span>min <var>Z</var> =</span>
              <span class="agy-sum">∑</span><sub><i>j</i>∈<i>F</i></sub>
              <span><var>f</var><sub><i>j</i></sub> <var>y</var><sub><i>j</i></sub> +</span>
              <span class="agy-sum">∑</span><sub><i>i</i>∈<i>P</i></sub>
              <span class="agy-sum">∑</span><sub><i>j</i>∈<i>F</i></sub>
              <span><var>c</var><sub><i>ij</i></sub> <var>x</var><sub><i>ij</i></sub></span>
            </div>
          </div>
          <p style="font-size:12.5px;color:#a1a1aa;margin:4px 0 0 0;">Sujeto a: cada origen <i>i</i> asignado a exactamente 1 centro abierto, oferta asignada ≤ <code>capacidad_frio_kg</code> (265.75 ≤ 5,000 kg) y <code>t_ij ≤ 4.0 h</code>.</p>
        </div>
      </div>

      <div class="agy-plinth-card">
        <h3>Desglose Tramo a Tramo de las 7 Rutas hacia ★ F02</h3>
        <p style="font-size:12.8px;margin-bottom:8px;">Pasa el cursor sobre una fila para inspeccionar el desglose completo de Dijkstra.</p>
        <div style="overflow-x:auto;">
          <table class="agy-flp-esc-table">
            <thead>
              <tr>
                <th>Orig.</th>
                <th>Ruta Óptima</th>
                <th>Suma tiempo_h</th>
                <th style="text-align:right;">Total</th>
                <th style="text-align:right;">Dist.</th>
              </tr>
            </thead>
            <tbody>${{filasF02}}</tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}}

function agyRenderizarSimulacionP05yFormulasFLP() {{
  const thWrap = document.getElementById("agy-flp-thermal-wrap");
  const det = AGY_DATA.flp_detallado || {{}};
  const sim = det.simulacion_p05;
  const f23 = det.arista_f02_f03;
  if (!thWrap || !sim) return;

  const chainHtml = (sim.pasos || []).map(p => {{
    let cls = "agy-flp-node-step";
    if (p.refrigerado) cls += " step-cold";
    else if (p.tipo === "D") cls += " step-danger";
    else if (p.tipo === "M") cls += " step-warn";
    return `
      <div class="${{cls}}" data-flp-step="${{p.paso}}">
        <div style="display:flex;justify-content:space-between;align-items:center;font-family:var(--font-mono);font-size:10.5px;color:#a1a1aa;">
          <span>PASO ${{p.paso}}</span>
          <span>${{p.temp_c}}°C</span>
        </div>
        <div style="font-size:14px;font-weight:700;color:#fff;margin:3px 0 2px 0;">${{p.nodo_id}} <span style="font-size:11px;color:#a1a1aa;font-weight:500;">(${{p.tipo}})</span></div>
        <div style="font-family:var(--font-mono);font-size:12px;color:#e4e4e7;">${{p.kg_actual.toFixed(3)}} kg</div>
        <div style="font-family:var(--font-mono);font-size:10.8px;color:${{p.refrigerado ? '#34d399' : (p.tipo === 'D' ? '#f87171' : '#fbbf24')}};margin-top:3px;">
          ${{p.paso === 0 ? 'Inicio 30 °C' : (p.refrigerado ? `Δ${{p.delta_det_pct.toFixed(2)}}% · Reset 0°C` : `Δ${{p.delta_det_pct.toFixed(2)}}% · ${{p.efecto}}`)}}
        </div>
      </div>
    `;
  }}).join("");

  const f23Block = f23 && f23.peso_combinado !== undefined ? `
    <div class="agy-flp-formula-card agy-tip-target" data-flp-tip="edge_peso">
      <div class="agy-flp-formula-head">
        <span class="agy-flp-formula-title">4. Peso Multicriterio — Sustitución Real Arista F02 → F03</span>
        <span class="agy-flp-formula-src">arista.py (ADR-02)</span>
      </div>
      <div class="agy-vm-math-stack">
        <div class="agy-vm-math-row">
          <span><var>t</var><sub>ef</sub> = ${{f23.tiempo_h}} × ${{f23.factor_estado_via}} × ${{f23.factor_flujo}} + ${{f23.tiempo_espera_h}} = <strong>${{f23.tiempo_efectivo_h}} h</strong></span>
          <span class="agy-op">·</span>
          <span><var>C</var><sub>tot</sub> = <strong>$${{f23.costo_total_cop.toLocaleString()}} COP</strong></span>
        </div>
        <div class="agy-vm-math-row">
          <span><var>w</var><sub>F02→F03</sub> =</span>
          <span>0.25(<span class="agy-frac"><span class="agy-frac-num">${{f23.tiempo_efectivo_h}}</span><span class="agy-frac-den">48</span></span>) + 0.20(<span class="agy-frac"><span class="agy-frac-num">453k</span><span class="agy-frac-den">5M</span></span>) + 0.20(${{f23.riesgo_combinado}}) + 0.10(${{f23.p_norm.toFixed(4)}}) + 0.10(1−${{f23.confiabilidad_via}}) + 0.05(${{f23.prob_retraso}}) = <strong style="color:#fde68a;">${{f23.peso_combinado.toFixed(4)}}</strong></span>
        </div>
      </div>
    </div>
  ` : "";

  thWrap.innerHTML = `
    <div class="agy-bento-grid-2">
      <div>
        <div class="agy-flp-formula-card agy-tip-target" data-flp-tip="pipe_filtro">
          <div class="agy-flp-formula-head">
            <span class="agy-flp-formula-title">1. Optimización FLP (Cobertura + Costo Mínimo)</span>
            <span class="agy-flp-formula-src">facility_location.py</span>
          </div>
          <div class="agy-vm-math-stack">
            <div class="agy-vm-math-row">
              <span>min <var>Z</var> =</span>
              <span class="agy-sum">∑</span><sub><i>j</i>∈<i>F</i></sub>
              <span><var>f</var><sub><i>j</i></sub> <var>y</var><sub><i>j</i></sub> +</span>
              <span class="agy-sum">∑</span><sub><i>i</i>∈<i>P</i></sub>
              <span class="agy-sum">∑</span><sub><i>j</i>∈<i>F</i></sub>
              <span><var>c</var><sub><i>ij</i></sub> <var>x</var><sub><i>ij</i></sub></span>
            </div>
            <div class="agy-vm-math-row" style="font-size:13px;color:#a1a1aa;">
              <span>s.a. <var>d</var><sub>tiempo</sub>(<i>i</i>, <i>j</i>) ≤ 4.0 h &nbsp;∧&nbsp; ∑ <var>q</var><sub><i>i</i></sub> <var>x</var><sub><i>ij</i></sub> ≤ <var>K</var><sub><i>j</i></sub> <var>y</var><sub><i>j</i></sub></span>
            </div>
          </div>
        </div>

        <div class="agy-flp-formula-card agy-tip-target" data-flp-tip="const_deterioro">
          <div class="agy-flp-formula-head">
            <span class="agy-flp-formula-title">2. Deterioro Térmico FAO (30 °C, Zona D y Muelle M)</span>
            <span class="agy-flp-formula-src">cold_chain.py + geo_utils.py</span>
          </div>
          <div class="agy-vm-math-stack">
            <div class="agy-vm-math-row">
              <span>Δ<var>d</var> = <var>t</var><sub>tramo</sub> × 2.5%/h × <var>φ</var><sub><i>D</i></sub></span>
              <span class="agy-op">·</span>
              <span><var>φ</var><sub><i>D</i></sub> = 1.5 (en Zona D)</span>
            </div>
            <div class="agy-vm-math-row">
              <span><var>t</var><sub>tramo,M</sub> = <var>t</var><sub>arista</sub> + 0.25 h</span>
              <span class="agy-op">·</span>
              <span><var>kg</var><sub>nuevo</sub> = <var>kg</var><sub>actual</sub> · (1 − <span class="agy-frac"><span class="agy-frac-num">Δ<var>d</var></span><span class="agy-frac-den">100</span></span>)</span>
            </div>
          </div>
        </div>

        <div class="agy-flp-formula-card agy-tip-target" data-flp-tip="const_fijo_esc">
          <div class="agy-flp-formula-head">
            <span class="agy-flp-formula-title">3. Costo Total Mensual del Escenario</span>
            <span class="agy-flp-formula-src">scenario_evaluator.py</span>
          </div>
          <div class="agy-vm-math-stack">
            <div class="agy-vm-math-row">
              <span><var>C</var><sub>total</sub> = (<var>n</var><sub>centros</sub> × $5,000,000) + <var>C</var><sub>transporte</sub> + (<var>kg</var><sub>perd</sub> × $12,000)</span>
            </div>
          </div>
        </div>

        ${{f23Block}}
      </div>

      <div class="agy-plinth-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
          <div>
            <h3 style="margin:0;">Simulación Térmica Paso a Paso — Ruta Crítica P05</h3>
            <p style="font-size:12.8px;margin:4px 0 0 0;">Lote inicial: <strong>${{sim.kg_inicial.toFixed(1)}} kg</strong> (P05 Puerto Trujillo → V_PG1). <strong style="color:#c4b5fd;">Pasa el cursor por cada paso</strong> para ver su cálculo exacto.</p>
          </div>
          <span class="agy-delta-pill">✓ cumple_limite = ${{sim.cumple_limite ? 'True' : 'False'}}</span>
        </div>

        <div class="agy-flp-chain">${{chainHtml}}</div>

        <div class="agy-bento-grid-3" style="margin-bottom:0;">
          <div class="agy-plinth-card obsidian" style="padding:12px;">
            <div style="font-size:11px;color:#9ca3af;font-family:var(--font-mono);">MASA FINAL ENTREGADA</div>
            <div style="font-size:20px;font-weight:700;color:#34d399;font-family:var(--font-mono);margin-top:4px;">${{sim.kg_final.toFixed(3)}} kg</div>
            <div style="font-size:11.5px;color:#a1a1aa;">de ${{sim.kg_inicial.toFixed(1)}} kg iniciales</div>
          </div>
          <div class="agy-plinth-card obsidian" style="padding:12px;">
            <div style="font-size:11px;color:#9ca3af;font-family:var(--font-mono);">MERMA ACUMULADA</div>
            <div style="font-size:20px;font-weight:700;color:#fbbf24;font-family:var(--font-mono);margin-top:4px;">${{sim.perdida_kg.toFixed(3)}} kg (${{sim.perdida_pct.toFixed(2)}}%)</div>
            <div style="font-size:11.5px;color:#a1a1aa;">$${{Math.round(sim.costo_perdida_cop).toLocaleString()}} COP en merma</div>
          </div>
          <div class="agy-plinth-card obsidian" style="padding:12px;">
            <div style="font-size:11px;color:#9ca3af;font-family:var(--font-mono);">TIEMPO TOTAL / FINAL SIN FRÍO</div>
            <div style="font-size:20px;font-weight:700;color:#f5f5f7;font-family:var(--font-mono);margin-top:4px;">${{sim.tiempo_total_h.toFixed(2)}} h / ${{sim.tiempo_sin_frio_final_h.toFixed(2)}} h</div>
            <div style="font-size:11.5px;color:#34d399;">Reset en A03, A01 y F02 (≤ 4.0 h)</div>
          </div>
        </div>
      </div>
    </div>
  `;
}}

function agyRenderizarEscenariosYAuditoriaFLP(modo) {{
  const escWrap = document.getElementById("agy-flp-scenarios-wrap");
  const audWrap = document.getElementById("agy-flp-audit-wrap");
  const lista = modo === "PG" ? (AGY_DATA.escenarios_pg || []) : (AGY_DATA.escenarios_bog || []);

  if (escWrap && lista.length) {{
    const maxCosto = Math.max(...lista.map(e => e.costo_total_cop || 0), 1);
    const filas = lista.map((e, idx) => {{
      const isOpt = idx === 2;
      const pctBar = Math.max(4, Math.round((e.costo_total_cop / maxCosto) * 100));
      const barCol = isOpt ? "#10b981" : (idx === 4 ? "#ef4444" : "#8052ff");
      return `
        <tr class="esc-hoverable ${{isOpt ? 'esc-opt' : ''}}" data-flp-esc="${{modo}}|${{idx}}">
          <td>${{isOpt ? '★ ' : ''}}${{e.nombre}}</td>
          <td style="font-family:var(--font-mono);text-align:right;">${{e.cumplimiento_termico_pct.toFixed(1)}}%</td>
          <td style="font-family:var(--font-mono);text-align:right;">${{e.pct_perdida.toFixed(2)}}% (${{e.kg_perdido.toFixed(1)}} kg)</td>
          <td style="font-family:var(--font-mono);text-align:right;">$${{Math.round(e.costo_total_cop).toLocaleString()}}</td>
          <td style="font-family:var(--font-mono);text-align:right;">$${{Math.round(e.costo_por_kg_cop).toLocaleString()}}/kg</td>
          <td style="min-width:130px;">
            <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:999px;overflow:hidden;">
              <div style="height:100%;width:${{pctBar}}%;background:${{barCol}};border-radius:999px;"></div>
            </div>
          </td>
        </tr>
      `;
    }}).join("");

    escWrap.innerHTML = `
      <div class="agy-plinth-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px;">
          <div>
            <h3 style="margin:0;">Evaluación Comparativa de los 5 Escenarios (${{modo === "PG" ? "Destino Local: V_PG1" : "Corredor Nacional: V_BOG1"}})</h3>
            <p style="font-size:13px;margin:4px 0 0 0;">Pasa el cursor sobre cualquier escenario para ver el desglose de centros abiertos, costo fijo uniforme ($5M/planta) y fórmula exacta.</p>
          </div>
          <span class="agy-delta-pill">AHORRO SOLO F02 vs. 3 PLANTAS: −$10,000,000 COP/MES</span>
        </div>
        <div style="overflow-x:auto;">
          <table class="agy-flp-esc-table">
            <thead>
              <tr>
                <th>Escenario Evaluado</th>
                <th style="text-align:right;">Cobertura</th>
                <th style="text-align:right;">Merma</th>
                <th style="text-align:right;">Costo Total / mes</th>
                <th style="text-align:right;">Costo / kg</th>
                <th>Peso Relativo</th>
              </tr>
            </thead>
            <tbody>${{filas}}</tbody>
          </table>
        </div>
      </div>
    `;
  }}

  if (audWrap) {{
    audWrap.innerHTML = `
      <div class="agy-plinth-card" style="border-color:rgba(245,158,11,0.35);">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:10px;">
          <h3 style="margin:0;color:#fbbf24;">Notas Críticas de Auditoría para Sustentación (Transparencia del Código)</h3>
          <span class="agy-badge" style="color:#fbbf24;border-color:rgba(245,158,11,0.4);">SECCIÓN 5 · INFORME FLP</span>
        </div>
        <div class="agy-bento-grid-2" style="margin-bottom:0;">
          <div style="font-size:12.8px;color:#d4d4d8;line-height:1.55;">
            <p style="margin:0 0 8px 0;"><strong>1. Costo fijo $3M (CSV) vs. $5M (EvaluadorEscenarios):</strong> <code>facility_location.py</code> lee <code>nodo.costo_fijo</code> de <code>nodos_centros.csv</code> ($3.0M para F02), mientras <code>EvaluadorEscenarios</code> aplica la constante uniforme <code>COSTO_FIJO_CENTRO_FRIO = $5,000,000 COP/mes</code> por cada planta abierta (de ahí el ahorro exacto de $10M COP/mes entre 1 y 3 centros).</p>
            <p style="margin:0;"><strong>2. Tiempo P05→F02 (2.65 h vs. 2.10 h):</strong> La matriz FLP suma las aristas puras (<code>0.20+1.20+0.40+0.25+0.60 = 2.65 h</code>). En cambio, <code>dijkstra_con_cota_inocuidad</code> reinicia el reloj en acopios con frío (A03/A01), reportando un tramo máximo continuo sin frío de <code>2.10 h</code>.</p>
          </div>
          <div style="font-size:12.8px;color:#d4d4d8;line-height:1.55;">
            <p style="margin:0 0 8px 0;"><strong>3. Escenario «Sin cadena de frío» en EvaluadorEscenarios:</strong> Mantiene los nodos de acopio (A01–A03) y F02 como vértices de paso del grafo, por lo que <code>SimuladorCadenaFrio</code> conserva su abatimiento al pasar por ellos; el escenario cuantifica el costo base de transporte + merma ($523,133 local / $8,023,272 Bogotá) sin costo fijo de planta.</p>
            <p style="margin:0;"><strong>4. «Solo F03» hacia V_PG1 = 0%:</strong> Porque el grafo es dirigido y existe <code>F02 → F03</code> (hacia el occidente) sin arista de retorno <code>F03 → V_PG1</code>; por eso F03 es nodo de relevo hacia Bogotá y no reemplaza a F02.</p>
          </div>
        </div>
      </div>
    `;
  }}
}}

function agyCambiarDestinoEditorial(modo) {{
  agyModoDestinoFLP = modo;
  document.getElementById("agy-ed-dest-pg")?.classList.toggle("active", modo === "PG");
  document.getElementById("agy-ed-dest-bog")?.classList.toggle("active", modo === "BOG");
  agyRenderizarEscenariosYAuditoriaFLP(modo);
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
    if (targetArt) {{
      targetArt.classList.add("active");
      agyAnimarContadores(targetArt);
    }}
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

  const ptsCampo = AGY_DATA.puntos_campo || [];
  if (ptsCampo.length) {{
    const grpC = document.createElement("optgroup");
    grpC.label = `Puntos de Campo Conectados por Río (${{ptsCampo.length}})`;
    ptsCampo.forEach(pt => {{
      const o = document.createElement("option");
      o.value = `C:${{pt.id}}`;
      o.textContent = `${{pt.title || pt.id}} → ${{pt.destino_relevo_id || 'M03'}} (${{pt.distancia_tramo_rio_km || 0}} km río | ${{pt.condicion_cuesta_rio || 'Fluvial'}})`;
      grpC.appendChild(o);
    }});
    sel.appendChild(grpC);
  }}
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
  }} else if (tipo === "C") {{
    const pt = (AGY_DATA.puntos_campo || []).find(x => x.id === id);
    if (pt) agyInspeccionarPuntoCampo(pt);
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
        if (Array.isArray(pt.coords_latlon) && pt.coords_latlon.length >= 2) {{
          const ptsRuta = pt.coords_latlon.map(c => {{
            const pr = agyProyectarSIG(c[1], c[0], W, H);
            return `${{pr.x.toFixed(1)}},${{pr.y.toFixed(1)}}`;
          }}).join(" ");
          const polyRuta = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
          polyRuta.setAttribute("points", ptsRuta);
          polyRuta.setAttribute("fill", "none");
          const esAguasAbajo = String(pt.condicion_cuesta_rio || "").includes("Aguas abajo");
          polyRuta.setAttribute("stroke", esAguasAbajo ? "#00b4d8" : "#ffb829");
          polyRuta.setAttribute("stroke-width", "0.85");
          polyRuta.setAttribute("stroke-dasharray", "3 2");
          polyRuta.setAttribute("stroke-opacity", "0.78");
          layerField.appendChild(polyRuta);
        }}
        const p = agyProyectarSIG(pt.longitud, pt.latitud, W, H);
        const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        dot.setAttribute("cx", p.x.toFixed(1));
        dot.setAttribute("cy", p.y.toFixed(1));
        dot.setAttribute("r", "2.1");
        dot.setAttribute("fill", AGY_COLORES_NODO[pt.categoria] || "#9a9a9a");
        dot.setAttribute("fill-opacity", "0.92");
        dot.setAttribute("stroke", "#000000");
        dot.setAttribute("stroke-width", "0.45");
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

  let guiaConstruccion = `
    <div class="agy-effect-box">
      <strong>Unión en Grafo Mixto Multimodal (25 V · 41 E):</strong><br>
      1.º Se creó <code>GrafoLogistico</code> (Lista de Adyacencia Hash O(1)).<br>
      2.º Se tipificaron los 25 vértices (<code>P, M, A, F, D, H, V</code>).<br>
      3.º Se soldaron el Grafo Fluvial (10 aristas) y el Grafo Vial (31 aristas) usando los Muelles (<code>M</code>) y Acopios (<code>A</code>) como nodos puente.<br>
      <button class="agy-modal-trigger" onclick="agyAbrirModalConstruccionSubgrafo()">+ Ver Analogía y Defensa Completa</button>
    </div>
  `;
  if (agySlideActual === 2) {{
    guiaConstruccion = `
      <div class="agy-effect-box">
        <strong>Construcción del Subgrafo Vial (19 V · 31 E):</strong><br>
        1.º Se cargaron los nodos terrestres (<code>M, A, F, D, H, V</code>).<br>
        2.º Se clasificaron 12 vías de <code>acceso_local</code> (trochas) y 19 vías <code>terrestres</code> (Ruta Nacional 40).<br>
        3.º Se trazó la geometría real con <strong>OSRM</strong> persistida en <code>rutas_cache.json</code>.<br>
        <button class="agy-modal-trigger" onclick="agyAbrirModalConstruccionSubgrafo()">+ Por qué OSRM y Cómo se Construyó</button>
      </div>
    `;
  }} else if (agySlideActual === 3) {{
    guiaConstruccion = `
      <div class="agy-effect-box">
        <strong>Construcción del Grafo de Ríos (rios.shp → 10 Rutas):</strong><br>
        1.º Se discretizó <code>rios.shp</code> (IGAC) en <strong>12,413 vértices</strong> y <strong>24,790 segmentos Haversine</strong>.<br>
        2.º Se indexó en una <strong>Rejilla Hash de 447 celdas</strong> (0.05°×0.05°) para proyectar nodos <code>P</code> al cauce en O(1).<br>
        3.º Un Dijkstra fluvial propio trazó las 10 rutas navegables reales.<br>
        <button class="agy-modal-trigger" onclick="agyAbrirModalConstruccionSubgrafo()">+ Por qué Doble Grafo y Defensa</button>
      </div>
    `;
  }}

  document.getElementById("agy-insp-body").innerHTML = `
    ${{guiaConstruccion}}

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
      <h4 class="agy-prop-h4-btn" role="button" tabindex="0" onclick="agyAbrirModalVariables('nodo', 1)" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();agyAbrirModalVariables('nodo', 1);}}" title="Clic para ver explicación y fórmulas de estas variables"><span>1. Identidad y Coordenadas WGS84</span><span class="agy-prop-h4-pill">Nodo · ?</span></h4>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_id"><span class="agy-prop-key">Identificador</span><span class="agy-prop-val">${{n.id}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_nombre"><span class="agy-prop-key">Nombre</span><span class="agy-prop-val">${{n.nombre}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_tipo"><span class="agy-prop-key">Rol logístico</span><span class="agy-prop-val">${{n.tipo}} · ${{n.tipo_descripcion}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_coords"><span class="agy-prop-key">Coordenadas GPS</span><span class="agy-prop-val">${{n.latitud.toFixed(4)}}°N, ${{n.longitud.toFixed(4)}}°W</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_tipo_via"><span class="agy-prop-key">Modos conectados</span><span class="agy-prop-val">${{(n.modos_conectados || []).join(" + ") || "—"}}</span></div>
    </div>

    <div class="agy-prop-group">
      <h4 class="agy-prop-h4-btn" role="button" tabindex="0" onclick="agyAbrirModalVariables('nodo', 2)" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();agyAbrirModalVariables('nodo', 2);}}" title="Clic para ver explicación y fórmulas de estas variables"><span>2. Capacidades, Producción y Costos</span><span class="agy-prop-h4-pill">Métricas · ?</span></h4>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_produccion"><span class="agy-prop-key">Producción diaria</span><span class="agy-prop-val">${{n.produccion_kg_dia.toFixed(2)}} kg/día</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_capacidad"><span class="agy-prop-key">Capacidad frigorífica</span><span class="agy-prop-val">${{n.capacidad_frio_kg.toLocaleString()}} kg</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_costo_fijo"><span class="agy-prop-key">Costo fijo mensual</span><span class="agy-prop-val">$${{n.costo_fijo.toLocaleString()}} COP</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_metadata"><span class="agy-prop-key">Demanda comercial</span><span class="agy-prop-val">${{n.metadata.demanda_kg_dia || 0}} kg/día</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_metadata"><span class="agy-prop-key">Cuenca hidrográfica</span><span class="agy-prop-val">${{n.metadata.rio || "—"}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_metadata"><span class="agy-prop-key">Pescadores censados</span><span class="agy-prop-val">${{n.metadata.pescadores || "—"}}</span></div>
    </div>

    <div class="agy-prop-group">
      <h4 class="agy-prop-h4-btn" role="button" tabindex="0" onclick="agyAbrirModalVariables('nodo', 3)" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();agyAbrirModalVariables('nodo', 3);}}" title="Clic para ver explicación y fórmulas de estas variables"><span>3. Infraestructura y Restricciones</span><span class="agy-prop-h4-pill">Estado · ?</span></h4>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_energia"><span class="agy-prop-key">Energía eléctrica</span><span class="agy-prop-val">${{n.tiene_energia ? "Sí (Continua)" : "No disponible"}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_agua"><span class="agy-prop-key">Agua potable</span><span class="agy-prop-val">${{n.tiene_agua ? "Sí" : "No"}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_muelle"><span class="agy-prop-key">Infraestructura muelle</span><span class="agy-prop-val">${{n.tiene_muelle ? "Sí" : "No"}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_restriccion"><span class="agy-prop-key">Restricción ambiental</span><span class="agy-prop-val">${{n.restriccion_ambiental ? "Hábitat protegido (H)" : "Sin restricción"}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="node_es_candidato"><span class="agy-prop-key">Elegible FLP</span><span class="agy-prop-val">${{n.es_candidato_frio ? "Candidato válido" : "No elegible"}}</span></div>
    </div>

    <div class="agy-prop-group">
      <h4 class="agy-prop-h4-btn" role="button" tabindex="0" onclick="agyAbrirModalVariables('nodo', 4)" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();agyAbrirModalVariables('nodo', 4);}}" title="Clic para ver explicación y fórmulas de conectividad"><span>4. Conectividad en Lista de Adyacencia</span><span class="agy-prop-h4-pill">deg⁻ / deg⁺ · ?</span></h4>
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
      <h4 class="agy-prop-h4-btn" role="button" tabindex="0" onclick="agyAbrirModalVariables('arista', 0)" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();agyAbrirModalVariables('arista', 0);}}" title="Clic para ver la fórmula completa del peso multicriterio"><span>Peso Multicriterio</span><span class="agy-prop-h4-pill"><span class="agy-tex"><var>w</var><sub><i>ij</i></sub> <span class="agy-op">=</span> ${{a.peso_combinado.toFixed(4)}}</span> · ?</span></h4>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span><span class="agy-tex"><var>&alpha;</var> &middot; <var>t&#770;</var><sub><i>ij</i></sub></span> Tiempo efectivo (0.25)</span><span>+${{d.t_term.toFixed(4)}} (${{pct(d.t_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.t_term)}}%"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span><span class="agy-tex"><var>&beta;</var> &middot; <var>c&#770;</var><sub><i>ij</i></sub></span> Costo total c/peajes (0.20)</span><span>+${{d.c_term.toFixed(4)}} (${{pct(d.c_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.c_term)}}%"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span><span class="agy-tex"><var>&gamma;</var> &middot; <var>r</var><sub>comb</sub></span> Riesgo combinado (0.20)</span><span>+${{d.r_term.toFixed(4)}} (${{pct(d.r_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.r_term)}}%"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span><span class="agy-tex"><var>&delta;</var> &middot; <var>p&#770;</var><sub><i>ij</i></sub></span> Pérdida esperada (0.10)</span><span>+${{d.p_term.toFixed(4)}} (${{pct(d.p_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.p_term)}}%"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span><span class="agy-tex"><var>&epsilon;</var> &middot; (1 <span class="agy-op">&minus;</span> <var>q</var><sub><i>ij</i></sub>)</span> Incertidumbre (0.10)</span><span>+${{d.conf_term.toFixed(4)}} (${{pct(d.conf_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.conf_term)}}%"></div></div>
      </div>
      <div class="agy-bar-wrap">
        <div class="agy-bar-label"><span><span class="agy-tex"><var>&zeta;</var> &middot; <var>P</var><sub>ret</sub></span> Prob. de retraso (0.05)</span><span>+${{d.ret_term.toFixed(4)}} (${{pct(d.ret_term)}}%)</span></div>
        <div class="agy-bar-bg"><div class="agy-bar-fill" style="width:${{pct(d.ret_term)}}%"></div></div>
      </div>
    </div>

    <div class="agy-prop-group">
      <h4 class="agy-prop-h4-btn" role="button" tabindex="0" onclick="agyAbrirModalVariables('arista', 1)" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();agyAbrirModalVariables('arista', 1);}}" title="Clic para ver explicación y fórmulas de cinemática y tiempo"><span>1. Distancia, Tiempo y Velocidad</span><span class="agy-prop-h4-pill">Cinemática · ?</span></h4>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_distancia"><span class="agy-prop-key">Distancia del tramo</span><span class="agy-prop-val">${{a.distancia_km.toFixed(2)}} km</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_tiempo"><span class="agy-prop-key">Tiempo base</span><span class="agy-prop-val">${{a.tiempo_h.toFixed(2)}} h</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_flujo"><span class="agy-prop-key">Flujo vehicular</span><span class="agy-prop-val">${{a.flujo_vehicular}} (×${{a.factor_flujo}})</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_espera"><span class="agy-prop-key">Tiempo de espera</span><span class="agy-prop-val">${{a.tiempo_espera_h.toFixed(2)}} h</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_tiempo_ef"><span class="agy-prop-key">Tiempo efectivo</span><span class="agy-prop-val">${{a.tiempo_efectivo_h.toFixed(3)}} h</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_tiempo_ef"><span class="agy-prop-key">Velocidad efectiva</span><span class="agy-prop-val">${{a.velocidad_efectiva_kmh.toFixed(1)}} km/h</span></div>
    </div>

    <div class="agy-prop-group">
      <h4 class="agy-prop-h4-btn" role="button" tabindex="0" onclick="agyAbrirModalVariables('arista', 2)" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();agyAbrirModalVariables('arista', 2);}}" title="Clic para ver explicación y fórmulas de costos y peajes"><span>2. Costos Logísticos y Peajes</span><span class="agy-prop-h4-pill">COP · ?</span></h4>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_costo_base"><span class="agy-prop-key">Flete base</span><span class="agy-prop-val">$${{a.costo_cop.toLocaleString()}} COP</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_peajes"><span class="agy-prop-key">Casetas de peaje</span><span class="agy-prop-val">${{a.peajes}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_peajes"><span class="agy-prop-key">Costo de peajes</span><span class="agy-prop-val">$${{a.costo_peajes_cop.toLocaleString()}} COP</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_costo_total"><span class="agy-prop-key">Costo total</span><span class="agy-prop-val">$${{a.costo_total_cop.toLocaleString()}} COP</span></div>
    </div>

    <div class="agy-prop-group">
      <h4 class="agy-prop-h4-btn" role="button" tabindex="0" onclick="agyAbrirModalVariables('arista', 3)" onkeydown="if(event.key==='Enter'||event.key===' '){{event.preventDefault();agyAbrirModalVariables('arista', 3);}}" title="Clic para ver explicación y fórmulas de riesgo y pérdida"><span>3. Estado Vial, Riesgo y Deterioro</span><span class="agy-prop-h4-pill">20 Variables · ?</span></h4>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_tipo_via"><span class="agy-prop-key">Modalidad de vía</span><span class="agy-prop-val">${{a.tipo_via}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_estado_via"><span class="agy-prop-key">Estado superficial</span><span class="agy-prop-val">${{a.estado_via}} (×${{a.factor_estado_via}})</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_confiabilidad"><span class="agy-prop-key">Confiabilidad vial</span><span class="agy-prop-val">${{a.confiabilidad_via.toFixed(2)}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_retraso"><span class="agy-prop-key">Probabilidad de retraso</span><span class="agy-prop-val">${{a.prob_retraso.toFixed(2)}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_riesgos"><span class="agy-prop-key">Riesgo térmico</span><span class="agy-prop-val">${{a.riesgo_termico.toFixed(2)}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_riesgos"><span class="agy-prop-key">Riesgo de accidente</span><span class="agy-prop-val">${{a.riesgo_accidente.toFixed(2)}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_riesgos"><span class="agy-prop-key">Riesgo de derrumbe</span><span class="agy-prop-val">${{a.riesgo_derrumbe.toFixed(2)}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_riesgos"><span class="agy-prop-key">Riesgo combinado</span><span class="agy-prop-val">${{a.riesgo_combinado.toFixed(4)}}</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_perdida"><span class="agy-prop-key">Pérdida esperada (L_ij)</span><span class="agy-prop-val">${{a.perdida_esperada.toFixed(2)}}%</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_capacidad"><span class="agy-prop-key">Capacidad / Peso máx.</span><span class="agy-prop-val">${{a.capacidad_kg}} kg · ${{a.restriccion_peso_ton}} t</span></div>
      <div class="agy-prop-row agy-tip-target" data-flp-tip="edge_capacidad"><span class="agy-prop-key">Restricción horaria</span><span class="agy-prop-val">${{a.horario_restriccion}}</span></div>
    </div>
  `;
}}

function agyInspeccionarPuntoCampo(pt) {{
  agySeleccion = {{ tipo: "campo", id: pt.id }};
  document.getElementById("agy-insp-title").textContent = `${{pt.title || pt.id}}`;
  const badge = document.getElementById("agy-insp-badge");
  badge.textContent = `${{pt.tipo || pt.categoria}} · ${{pt.cota_origen_msnm || 150}} m s.n.m.`;
  badge.style.color = AGY_COLORES_NODO[pt.categoria] || "#2997ff";

  document.getElementById("agy-insp-body").innerHTML = `
    <div class="agy-effect-box">
      <strong>🚤 Conexión por Red Fluvial Real (rios.shp — Lancha 40 HP):</strong><br>
      Ruta en lancha: <strong>${{pt.id}} ──río──► ${{pt.destino_relevo_id || 'M03'}} ──► M03 (Puerto Gaitán)</strong>.<br>
      Condición hidráulica: <strong style="color:var(--color-saffron-spark);">${{pt.condicion_cuesta_rio || 'Fluvial'}}</strong>.
    </div>

    <div class="agy-prop-group">
      <h4><span>1. Recorrido por Red Fluvial (rios.shp)</span><span>Geometría Real</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Punto de origen</span><span class="agy-prop-val">${{pt.title || pt.id}} (${{pt.tipo}})</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Coordenadas DMS</span><span class="agy-prop-val">${{pt.lat_dms}}, ${{pt.lon_dms}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Siguiente relevo fluvial</span><span class="agy-prop-val">${{pt.destino_relevo_nombre || 'M03 Puerto Gaitán'}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Distancia tramo por río</span><span class="agy-prop-val">${{pt.distancia_tramo_rio_km}} km (recta: ${{pt.distancia_recta_km}} km)</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Sinuosidad de meandros</span><span class="agy-prop-val">${{pt.indice_sinuosidad}}× (enlace al cauce: ${{pt.distancia_al_cauce_km}} km)</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Distancia fluvial total a M03</span><span class="agy-prop-val" style="color:var(--color-saffron-spark);">${{pt.distancia_total_a_m03_km}} km</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>2. Cuesta del Río, Pendiente y Lancha</span><span>Hidrología Real</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Régimen de cuesta</span><span class="agy-prop-val">${{pt.condicion_cuesta_rio}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Cota origen → M03</span><span class="agy-prop-val">${{pt.cota_origen_msnm}} → ${{pt.cota_m03_msnm}} m s.n.m. (${{pt.desnivel_a_m03_m >= 0 ? '+' : ''}}${{pt.desnivel_a_m03_m}} m)</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Pendiente hidráulica</span><span class="agy-prop-val">${{pt.pendiente_m_km}} m/km</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Velocidad corriente río</span><span class="agy-prop-val">${{pt.velocidad_corriente_kmh >= 0 ? '+' : ''}}${{pt.velocidad_corriente_kmh}} km/h</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Velocidad lancha (ida / retorno)</span><span class="agy-prop-val">${{pt.velocidad_lancha_ida_kmh}} km/h / ${{pt.velocidad_lancha_retorno_kmh}} km/h</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Esfuerzo motor fuera de borda</span><span class="agy-prop-val">${{pt.esfuerzo_motor}}</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Tiempo tramo / Total a M03</span><span class="agy-prop-val">${{pt.tiempo_tramo_h}} h / ${{pt.tiempo_total_a_m03_h}} h (${{pt.tiempo_total_a_m03_min}} min)</span></div>
    </div>

    <div class="agy-prop-group">
      <h4><span>3. Frío, Combustible e Inocuidad FAO</span><span>Lote ${{pt.lote_referencia_kg}} kg</span></h4>
      <div class="agy-prop-row"><span class="agy-prop-key">Gasolina lancha hasta M03</span><span class="agy-prop-val">${{pt.galones_gasolina}} gal ($${{(pt.costo_combustible_cop || 0).toLocaleString()}} COP)</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Hielo 1:1 requerido a bordo</span><span class="agy-prop-val">${{pt.hielo_requerido_1a1_kg}} kg hielo en escarcha</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Pérdida a 30 °C (sin hielo)</span><span class="agy-prop-val" style="color:var(--color-ember-accent);">${{pt.perdida_sin_hielo_pct}}%</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Pérdida a 0 °C (con hielo 1:1)</span><span class="agy-prop-val" style="color:#10b981;">${{pt.perdida_con_hielo_1a1_pct}}%</span></div>
      <div class="agy-prop-row"><span class="agy-prop-key">Diagnóstico FAO (4.0 h)</span><span class="agy-prop-val" style="color:${{pt.requiere_hielo_abordo ? 'var(--color-ember-accent)' : '#10b981'}};">${{pt.estado_fao_sin_hielo}}</span></div>
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
      agyCerrarModalDetalle();
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

  window.addEventListener("message", (ev) => {{
    if (!ev.data || typeof ev.data !== "object") return;
    if (ev.data.type === "agyAbrirCalculoFLP") {{
      agyCambiarSlide(6, false);
      if (typeof agySeleccionarCandidatoFLP === "function") {{
        agySeleccionarCandidatoFLP(ev.data.candidato || "F02");
      }}
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
