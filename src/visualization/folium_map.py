"""
folium_map.py — Mapa interactivo HTML con Folium.

Genera un mapa web interactivo que se abre en el navegador,
sin necesidad de cuentas externas ni servicios de pago.

Las rutas siguen las carreteras y ríos reales obtenidos
desde OSRM (Open Source Routing Machine) y el shapefile de ríos.

Capas generadas:
  1. Nodos de la red — marcadores categorizados por tipo y color.
  2. Ruta óptima — polilínea roja siguiendo carreteras reales.
  3. Todas las aristas — conexiones siguiendo vías reales.
  4. Ríos — capa GeoJSON de la red fluvial.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict

import folium
from folium import plugins

from ..models.grafo_logistico import GrafoLogistico
from ..algorithms.dijkstra import dijkstra
from ..utils.data_loader import OUTPUT_DIR, SHP_DIR
from ..utils.constants import TipoNodo, TIEMPO_MAX_SIN_FRIO_H, COSTO_FIJO_CENTRO_FRIO

# Ruta al GeoJSON de puntos de campo
_GEOJSON_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "geojson"


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE COLORES Y ICONOS POR TIPO DE NODO
# ═══════════════════════════════════════════════════════════════════

COLORES_FELT = {
    TipoNodo.PESCA:     "#2196F3",  # Azul
    TipoNodo.ACOPIO:    "#FF9800",  # Naranja
    TipoNodo.FRIO:      "#00BCD4",  # Cyan
    TipoNodo.MUELLE:    "#9C27B0",  # Púrpura
    TipoNodo.VENTA:     "#E91E63",  # Rosa
    TipoNodo.HABITAT:   "#4CAF50",  # Verde
    TipoNodo.DETERIORO: "#F44336",  # Rojo
}

ICONOS_FOLIUM = {
    TipoNodo.PESCA:     ("fish",      "blue"),
    TipoNodo.ACOPIO:    ("warehouse", "orange"),
    TipoNodo.FRIO:      ("snowflake", "cadetblue"),
    TipoNodo.MUELLE:    ("anchor",    "purple"),
    TipoNodo.VENTA:     ("shop",      "pink"),
    TipoNodo.HABITAT:   ("leaf",      "green"),
    TipoNodo.DETERIORO: ("warning",   "red"),
}


# ═══════════════════════════════════════════════════════════════════
#  GENERADOR DE MAPA INTERACTIVO
# ═══════════════════════════════════════════════════════════════════

def generar_mapa_interactivo(
    grafo: GrafoLogistico,
    ruta_optima: Optional[List[str]] = None,
    nombre_ruta: str = "Ruta Óptima Dijkstra",
    nombre_archivo: str = "mapa_interactivo.html",
    abrir_navegador: bool = False,
    geometrias_reales: Optional[Dict[str, list]] = None,
    itinerarios: Optional[list] = None,
) -> str:
    """
    Genera un mapa HTML interactivo con Folium.

    Parameters
    ----------
    grafo : GrafoLogistico
    ruta_optima : list[str] | None — IDs de nodos de la ruta óptima.
    nombre_ruta : str — nombre descriptivo para la ruta.
    nombre_archivo : str — nombre del archivo HTML de salida.
    abrir_navegador : bool — si abrir automáticamente en el navegador.
    geometrias_reales : dict | None — caché de geometrías reales
        {"{origen}-{destino}": [[lon, lat], ...]}

    Returns
    -------
    str : ruta absoluta del archivo HTML generado.
    """
    geo_cache = geometrias_reales or {}
    if not geo_cache:
        # Sin geometrías explícitas: usar la caché de vías/cauces reales
        ruta_cache = OUTPUT_DIR / "rutas_cache.json"
        if ruta_cache.exists():
            try:
                with open(ruta_cache, "r", encoding="utf-8") as f:
                    geo_cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                geo_cache = {}

    # ── Calcular centro del mapa ─────────────────────────────
    lats = [n.latitud for n in grafo.nodos if n.latitud != 0]
    lons = [n.longitud for n in grafo.nodos if n.longitud != 0]

    if not lats or not lons:
        centro = [4.31, -72.08]  # Puerto Gaitán por defecto
    else:
        centro = [sum(lats) / len(lats), sum(lons) / len(lons)]

    # ── Crear mapa base ──────────────────────────────────────
    mapa = folium.Map(
        location=centro,
        zoom_start=9,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Agregar capas base adicionales
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(mapa)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satélite (Esri)",
    ).add_to(mapa)

    itinerarios = list(itinerarios or [])

    # ── Capa de ríos reales (siempre visible) ────────────────
    _agregar_capa_rios(mapa)

    # ── Capa de puntos de campo (GeoJSON externo) ─────────
    _agregar_puntos_campo(mapa)

    # ── Capa de aristas (todas las conexiones) ───────────────
    capa_aristas = folium.FeatureGroup(name="🔗 Conexiones del Grafo", show=True)
    colores_tipo_estado = {
        ("terrestre", "buena"): ("#2E7D32", 3.5, None),
        ("terrestre", "regular"): ("#EF6C00", 3.0, None),
        ("terrestre", "mala"): ("#C62828", 2.5, "6 4"),
        ("acceso_local", "buena"): ("#546E7A", 2.5, None),
        ("acceso_local", "regular"): ("#78909C", 2.2, "4 4"),
        ("acceso_local", "mala"): ("#8D6E63", 2.0, "3 5"),
        ("fluvial", "buena"): ("#1565C0", 3.0, "8 6"),
        ("fluvial", "regular"): ("#42A5F5", 2.5, "8 6"),
        ("fluvial", "mala"): ("#0D47A1", 2.2, "4 6"),
    }
    for arista in grafo.aristas:
        n_o = grafo.obtener_nodo(arista.origen)
        n_d = grafo.obtener_nodo(arista.destino)
        if not n_o or not n_d:
            continue

        color_via, peso_via, dash_via = colores_tipo_estado.get(
            (arista.tipo_via, arista.estado_via),
            ("#616161", 2.5, None),
        )

        popup_html = (
            f"<b>{n_o.nombre} ({arista.origen}) → {n_d.nombre} ({arista.destino})</b><br>"
            f"🛣️ <b>tipo_via:</b> {arista.tipo_via} | <b>estado_via:</b> {arista.estado_via}<br>"
            f"📏 <b>distancia_km:</b> {arista.distancia_km:.2f} km<br>"
            f"⏱️ <b>tiempo_h:</b> {arista.tiempo_h:.2f} h | "
            f"<b>tiempo_espera_h:</b> {arista.tiempo_espera_h:.2f} h | "
            f"<b>tiempo_efectivo_h:</b> {arista.tiempo_efectivo_h:.2f} h<br>"
            f"💰 <b>costo_cop:</b> ${arista.costo_cop:,.0f} | "
            f"<b>peajes:</b> {arista.peajes} (${arista.costo_peajes_cop:,.0f}) | "
            f"<b>costo_total_cop:</b> ${arista.costo_total_cop:,.0f}<br>"
            f"🚗 <b>flujo_vehicular:</b> {arista.flujo_vehicular} | "
            f"<b>confiabilidad_via:</b> {arista.confiabilidad_via:.2f} | "
            f"<b>prob_retraso:</b> {arista.prob_retraso:.2f}<br>"
            f"🌡️ <b>riesgo_termico:</b> {arista.riesgo_termico:.2f} | "
            f"<b>riesgo_accidente:</b> {arista.riesgo_accidente:.2f} | "
            f"<b>riesgo_derrumbe:</b> {arista.riesgo_derrumbe:.2f} | "
            f"<b>riesgo_combinado:</b> {arista.riesgo_combinado:.3f}<br>"
            f"⚖️ <b>capacidad_kg:</b> {arista.capacidad_kg:,.0f} kg | "
            f"<b>restriccion_peso_ton:</b> {arista.restriccion_peso_ton} t | "
            f"<b>horario_restriccion:</b> {arista.horario_restriccion}<br>"
            f"📉 <b>perdida_esperada:</b> {arista.perdida_esperada:.2f}% | "
            f"⚖️ <b>peso_default:</b> {arista.peso_default:.4f}"
        )

        # Usar geometría real si disponible, si no línea recta
        clave = f"{arista.origen}-{arista.destino}"
        geo_coords = geo_cache.get(clave)
        if geo_coords and len(geo_coords) > 1:
            # geo_coords es [[lon, lat], ...] → convertir a [[lat, lon], ...]
            locations = [[c[1], c[0]] for c in geo_coords]
        else:
            locations = [[n_o.latitud, n_o.longitud],
                         [n_d.latitud, n_d.longitud]]

        folium.PolyLine(
            locations=locations,
            color=color_via,
            weight=peso_via,
            opacity=0.75,
            dash_array=dash_via,
            popup=folium.Popup(popup_html, max_width=360),
            tooltip=(
                f"{arista.origen}→{arista.destino} ({arista.tipo_via}/{arista.estado_via}): "
                f"{arista.distancia_km} km | {arista.tiempo_efectivo_h:.2f}h | w={arista.peso_default:.3f}"
            ),
        ).add_to(capa_aristas)

    capa_aristas.add_to(mapa)

    # ── Capa de ruta óptima ──────────────────────────────────
    if ruta_optima and len(ruta_optima) >= 2:
        capa_ruta = folium.FeatureGroup(name=f"🛤️ {nombre_ruta}")
        dist_total = 0.0
        tiempo_total = 0.0

        # Construir coordenadas de la ruta usando geometrías reales
        coordenadas_ruta = []
        for i in range(len(ruta_optima) - 1):
            clave = f"{ruta_optima[i]}-{ruta_optima[i + 1]}"
            ar = grafo.obtener_arista(ruta_optima[i], ruta_optima[i + 1])
            if ar:
                dist_total += ar.distancia_km
                tiempo_total += ar.tiempo_h

            geo_seg = geo_cache.get(clave)
            if geo_seg and len(geo_seg) > 1:
                segmento = [[c[1], c[0]] for c in geo_seg]
            else:
                n_o = grafo.obtener_nodo(ruta_optima[i])
                n_d = grafo.obtener_nodo(ruta_optima[i + 1])
                segmento = []
                if n_o:
                    segmento.append([n_o.latitud, n_o.longitud])
                if n_d:
                    segmento.append([n_d.latitud, n_d.longitud])

            # Evitar duplicar punto de conexión
            if coordenadas_ruta and segmento:
                coordenadas_ruta.extend(segmento[1:])
            else:
                coordenadas_ruta.extend(segmento)

        # Línea principal de la ruta
        folium.PolyLine(
            locations=coordenadas_ruta,
            color="#D32F2F",
            weight=5,
            opacity=0.9,
            popup=folium.Popup(
                f"<b>{nombre_ruta}</b><br>"
                f"📏 {dist_total:.1f} km<br>"
                f"⏱️ {tiempo_total:.2f} horas<br>"
                f"📍 {' → '.join(ruta_optima)}",
                max_width=300,
            ),
            tooltip=f"{nombre_ruta}: {dist_total:.1f} km",
            dash_array="",
        ).add_to(capa_ruta)

        # Decorador con flechas de dirección
        try:
            plugins.AntPath(
                locations=coordenadas_ruta,
                color="#D32F2F",
                weight=3,
                opacity=0.6,
                delay=1000,
            ).add_to(capa_ruta)
        except Exception:
            pass  # AntPath es decorativo, no crítico

        capa_ruta.add_to(mapa)

    # ── Capa de nodos (marcadores) ───────────────────────────
    acopio_info = _resumen_acopios(itinerarios)
    capa_frio_js_name = None
    frio_markers_meta = []

    for tipo_code in [TipoNodo.PESCA, TipoNodo.ACOPIO, TipoNodo.FRIO,
                      TipoNodo.VENTA]:
        nodos_tipo = grafo.nodos_por_tipo(tipo_code)
        if not nodos_tipo:
            continue

        desc = TipoNodo.descripcion(tipo_code)
        icono_nombre, icono_color = ICONOS_FOLIUM.get(
            tipo_code, ("circle", "gray")
        )
        if tipo_code == TipoNodo.FRIO:
            ids_frio = ", ".join(n.id for n in nodos_tipo)
            capa_nombre = f"❄️ Centros de Frío ({ids_frio})"
        else:
            capa_nombre = (
                f"{'🐟' if tipo_code == 'P' else '📦' if tipo_code == 'A' else '🏪' if tipo_code == 'V' else '⚓'} {desc}"
            )
        capa = folium.FeatureGroup(name=capa_nombre, show=True)

        for nodo in nodos_tipo:
            # Construir popup HTML rico
            popup_lines = [f"<b>{nodo.nombre}</b><br>"]
            if nodo.tipo == TipoNodo.FRIO and nodo.id == "F02":
                popup_lines.append(
                    "<span style='display:inline-block;background:#0288D1;color:#fff;"
                    "font-size:10px;font-weight:bold;padding:1px 6px;border-radius:4px;"
                    "margin:2px 0 4px 0;'>★ SELECCIÓN ÓPTIMA FLP (100% Cobertura)</span><br>"
                )
            popup_lines.append(f"<i>{desc}</i> ({nodo.id})<br>")

            if nodo.tipo == TipoNodo.PESCA:
                popup_lines.append(
                    f"🐟 Producción: {nodo.produccion_kg_dia} kg/día<br>"
                )
                if nodo.metadata.get("rio"):
                    popup_lines.append(
                        f"🌊 Río: {nodo.metadata['rio']}<br>"
                    )
                if nodo.metadata.get("pescadores"):
                    popup_lines.append(
                        f"👥 Pescadores: {nodo.metadata['pescadores']}<br>"
                    )
            elif nodo.tipo in (TipoNodo.FRIO, TipoNodo.ACOPIO):
                popup_lines.append(
                    f"❄️ Capacidad: {nodo.capacidad_frio_kg:,.0f} kg<br>"
                )
                popup_lines.append(
                    f"💰 Costo fijo: ${nodo.costo_fijo:,.0f} COP/mes<br>"
                )
                popup_lines.append(
                    f"⚡ Energía: {'Sí' if nodo.tiene_energia else 'No'} · "
                    f"💧 Agua: {'Sí' if nodo.tiene_agua else 'No'}<br>"
                )
                obs = nodo.metadata.get("observacion")
                if obs:
                    popup_lines.append(f"📝 {obs}<br>")
            elif nodo.tipo == TipoNodo.VENTA:
                demanda = nodo.metadata.get("demanda_kg_dia", 0)
                if demanda:
                    popup_lines.append(
                        f"📊 Demanda: {demanda} kg/día<br>"
                    )

            if nodo.tipo == TipoNodo.ACOPIO:
                info = acopio_info.get(nodo.id)
                if info:
                    pts = ", ".join(sorted(info["origenes"]))
                    popup_lines.append(
                        f"📥 <b>Recibe de:</b> {pts}<br>"
                        f"⚖️ <b>Flujo total:</b> {info['kg']:,.1f} kg/día<br>"
                    )
                else:
                    popup_lines.append("📥 Sin flujo asignado en los itinerarios<br>")

            popup_lines.append(
                f"📍 {nodo.latitud:.4f}, {nodo.longitud:.4f}"
            )

            if nodo.tipo == TipoNodo.FRIO:
                es_optimo = nodo.id == "F02"
                # Halo de área de cobertura frigorífica visible sobre el mapa satelital
                folium.Circle(
                    location=[nodo.latitud, nodo.longitud],
                    radius=2600 if es_optimo else 1600,
                    color="#0288D1" if es_optimo else "#00ACC1",
                    weight=2.2 if es_optimo else 1.5,
                    dash_array="" if es_optimo else "6 4",
                    fill=True,
                    fill_color="#29B6F6" if es_optimo else "#4DD0E1",
                    fill_opacity=0.18 if es_optimo else 0.12,
                    tooltip=f"❄️ Área de cobertura {nodo.nombre}",
                ).add_to(capa)

                # Marcador destacado con ícono de copo de nieve
                mk_frio = folium.Marker(
                    location=[nodo.latitud, nodo.longitud],
                    icon=folium.Icon(
                        color="darkblue" if es_optimo else "cadetblue",
                        icon="snowflake",
                        prefix="fa",
                    ),
                    popup=folium.Popup("".join(popup_lines), max_width=300),
                    tooltip=(
                        f"❄️ {nodo.id} ★ ÓPTIMO — {nodo.nombre} ({nodo.capacidad_frio_kg:,.0f} kg)"
                        if es_optimo
                        else f"❄️ {nodo.id} — {nodo.nombre} ({nodo.capacidad_frio_kg:,.0f} kg)"
                    ),
                    z_index_offset=900 if es_optimo else 700,
                )
                mk_frio.add_to(capa)
                frio_markers_meta.append({
                    "id": nodo.id,
                    "nombre": nodo.nombre.replace("Centro Frío ", ""),
                    "lat": nodo.latitud,
                    "lon": nodo.longitud,
                    "cap": int(nodo.capacidad_frio_kg),
                    "costo": int(nodo.costo_fijo),
                    "optimo": es_optimo,
                    "marker_var": mk_frio.get_name(),
                })
            else:
                # Determinar tamaño del marcador
                if nodo.tipo == TipoNodo.PESCA:
                    radio = max(6, min(nodo.produccion_kg_dia / 5, 15))
                elif nodo.tipo == TipoNodo.ACOPIO:
                    radio = max(8, min(nodo.capacidad_frio_kg / 400, 18))
                else:
                    radio = 10

                # Marcador circular con color del tipo
                folium.CircleMarker(
                    location=[nodo.latitud, nodo.longitud],
                    radius=radio,
                    color=COLORES_FELT.get(tipo_code, "#757575"),
                    fill=True,
                    fill_color=COLORES_FELT.get(tipo_code, "#757575"),
                    fill_opacity=0.8,
                    popup=folium.Popup("".join(popup_lines), max_width=280),
                    tooltip=f"{nodo.nombre} ({desc})",
                ).add_to(capa)

        capa.add_to(mapa)
        if tipo_code == TipoNodo.FRIO:
            capa_frio_js_name = capa.get_name()

    # ── Capa independiente: Nodos Candidatos a Centro de Frío (FLP) ──
    candidatos_nodos = grafo.candidatos_frio()
    capa_candidatos = folium.FeatureGroup(
        name=f"🎯 Candidatos a Centro de Frío ({len(candidatos_nodos)})",
        show=False,
    )
    candidatos_meta = []
    for cand in candidatos_nodos:
        es_opt = cand.id == "F02"
        estado_txt = (
            "★ SELECCIÓN ÓPTIMA FLP"
            if es_opt
            else ("Candidato Frío (F)" if cand.tipo == TipoNodo.FRIO else "Candidato Acopio con Energía (A)")
        )
        popup_cand = (
            f"<b>🎯 Candidato FLP: {cand.nombre}</b><br>"
            f"<span style='display:inline-block;background:{'#0288D1' if es_opt else '#7C3AED'};"
            f"color:#fff;font-size:10px;font-weight:bold;padding:1px 6px;border-radius:4px;"
            f"margin:2px 0 4px 0;'>{estado_txt}</span><br>"
            f"🆔 Nodo: <b>{cand.id}</b> ({TipoNodo.descripcion(cand.tipo)})<br>"
            f"❄️ Capacidad: {cand.capacidad_frio_kg:,.0f} kg<br>"
            f"💰 Costo fijo: ${cand.costo_fijo:,.0f} COP/mes<br>"
            f"⚡ Energía: Sí · 🌿 Sin restricción ambiental<br>"
            f"📍 {cand.latitud:.4f}, {cand.longitud:.4f}"
        )
        # Halo violeta discontinuo para distinguir todos los candidatos sin alterar los 3 centros F
        folium.Circle(
            location=[cand.latitud, cand.longitud],
            radius=2100,
            color="#7C3AED",
            weight=2.4,
            dash_array="7 5",
            fill=True,
            fill_color="#8B5CF6",
            fill_opacity=0.14,
            popup=folium.Popup(popup_cand, max_width=290),
            tooltip=f"🎯 Candidato {cand.id} — {cand.nombre} ({cand.capacidad_frio_kg:,.0f} kg)",
        ).add_to(capa_candidatos)

        # Etiqueta flotante limpia sobre cada nodo candidato
        badge_bg = "#6D28D9" if not es_opt else "#0288D1"
        badge_lbl = f"🎯 {cand.id} · {cand.nombre.replace('Centro Frío ', '').replace('Acopio ', '')}"
        folium.Marker(
            location=[cand.latitud, cand.longitud],
            icon=folium.DivIcon(
                icon_size=(165, 22),
                icon_anchor=(82, 34),
                html=(
                    f'<div style="display:inline-block;white-space:nowrap;'
                    f'background:{badge_bg};color:#fff;font-family:Arial,sans-serif;'
                    f'font-size:10px;font-weight:700;padding:2px 7px;border-radius:999px;'
                    f'border:1.5px solid #fff;box-shadow:0 2px 5px rgba(0,0,0,0.35);">'
                    f'{badge_lbl}</div>'
                ),
            ),
            popup=folium.Popup(popup_cand, max_width=290),
            z_index_offset=950,
        ).add_to(capa_candidatos)
        candidatos_meta.append({"id": cand.id, "lat": cand.latitud, "lon": cand.longitud})

    capa_candidatos.add_to(mapa)
    capa_candidatos_js_name = capa_candidatos.get_name()

    # ── Nodos M, D, H (capas propias con efecto) ─────────────
    _agregar_nodos_especiales(mapa, grafo, itinerarios)

    # ── Itinerarios por punto de pesca + flujo de acopio ─────
    if itinerarios:
        _agregar_itinerarios(mapa, grafo, itinerarios, geo_cache)

    # ── Leyenda y panel de itinerarios ───────────────────────
    _agregar_leyenda(mapa)
    if itinerarios:
        _agregar_panel_itinerarios(mapa, grafo, itinerarios)

    # ── Título, selector de Centros de Frío, Ficha Rápida FLP y ajuste anti-superposición ──
    map_js_var = mapa.get_name()
    frio_js_var = capa_frio_js_name or "null"
    cand_js_var = capa_candidatos_js_name or "null"
    frio_json = json.dumps(frio_markers_meta, ensure_ascii=False)
    cand_json = json.dumps(candidatos_meta, ensure_ascii=False)

    # Datos dinámicos para la ficha rápida FLP (Paso 1 de propuesta_flujo_visual_flp.md)
    puntos_pesca_flp = sorted(grafo.nodos_por_tipo(TipoNodo.PESCA), key=lambda x: x.id)
    oferta_total_kg = sum(p.produccion_kg_dia for p in puntos_pesca_flp)
    _provenance_map = {
        "F01": (
            "Manual +0.005° (~780 m) desde P01 ASOPESGA (commit 9af49ef)",
            "Cubre 4/7 (57.1%). P03, P04 y P05 no alcanzan F01 en el grafo dirigido (∞).",
        ),
        "F02": (
            "Cabecera urbana Puerto Gaitán (constants.py · 4.3128, -72.0826)",
            "Único candidato con cobertura 7/7 (100%) en ≤ 2.65 h (Dijkstra FLP). Ahorra $10M COP/mes vs. 3 plantas.",
        ),
        "F03": (
            "Cabecera Puerto López (constants.py · 126 km por Ruta Nacional 40)",
            "Cubre 6/7 (85.7%). P05→F03 tarda 5.15 h (> 4.0 h FAO) y no tiene retorno dirigido a V_PG1.",
        ),
        "A01": (
            "Mismas coordenadas que P03 San Miguel (commit 9af49ef)",
            "Cubre 3/7 (42.9%) del sur (P03, P04, P05). P04→A01 desvía por H01 (0.25 h).",
        ),
        "A02": (
            "Punto medio exacto entre P02 (4.33, -72.10) y P07 (4.34, -72.11)",
            "Cubre 4/7 (57.1%) del norte (P01, P02, P06, P07) en ≤ 0.35 h.",
        ),
    }
    quick_flp_map: Dict[str, dict] = {}
    for cand in candidatos_nodos:
        cubiertos = []
        tiempos_ok = []
        for p in puntos_pesca_flp:
            dist_t, cam_t = dijkstra(grafo, p.id, cand.id, criterio="tiempo_h", evitar_restringidos=True)
            if cam_t and dist_t <= TIEMPO_MAX_SIN_FRIO_H:
                cubiertos.append(p.id)
                tiempos_ok.append(dist_t)
        n_total = len(puntos_pesca_flp) or 1
        n_cub = len(cubiertos)
        pct_cub = round(n_cub / n_total * 100.0, 1)
        t_max = round(max(tiempos_ok), 2) if tiempos_ok else None
        uso_pct = round((oferta_total_kg / cand.capacidad_frio_kg) * 100.0, 1) if cand.capacidad_frio_kg > 0 else 0.0
        orig_txt, clave_txt = _provenance_map.get(
            cand.id,
            ("nodos_centros.csv", f"Candidato {cand.id} evaluado por Dijkstra (tiempo_h)."),
        )
        quick_flp_map[cand.id] = {
            "id": cand.id,
            "nombre": cand.nombre,
            "tipo": cand.tipo,
            "optimo": cand.id == "F02",
            "cobertura_n": n_cub,
            "cobertura_total": n_total,
            "cobertura_pct": pct_cub,
            "tiempo_max_h": t_max,
            "capacidad_kg": int(cand.capacidad_frio_kg),
            "uso_pct": uso_pct,
            "oferta_kg": round(oferta_total_kg, 2),
            "costo_fijo_csv": int(cand.costo_fijo),
            "costo_eval_esc": int(COSTO_FIJO_CENTRO_FRIO) if cand.tipo == TipoNodo.FRIO else int(cand.costo_fijo),
            "lat": cand.latitud,
            "lon": cand.longitud,
            "origen": orig_txt,
            "filtro": "tipo ∈ {'F','A'} ∧ tiene_energia=True ∧ ¬restriccion_ambiental",
            "clave": clave_txt,
            "cubiertos_ids": cubiertos,
        }
    quick_flp_json = json.dumps(quick_flp_map, ensure_ascii=False)

    botones_frio_html = ""
    for fm in frio_markers_meta:
        badge_opt = "★ " if fm["optimo"] else ""
        cls_btn = "frio-btn frio-btn-opt" if fm["optimo"] else "frio-btn"
        botones_frio_html += (
            f'<button type="button" class="{cls_btn}" '
            f'onclick="window.enfocarCentroFrio(\'{fm["id"]}\')" '
            f'title="Capacidad: {fm["cap"]:,} kg · Costo fijo CSV: ${fm["costo"]:,} COP/mes · Clic para abrir ficha FLP">'
            f'{badge_opt}{fm["id"]} · {fm["nombre"]} ({fm["cap"]:,} kg)</button>'
        )

    titulo_html = f"""
    <style>
      #folium-title-card {{
        position: fixed;
        top: 10px;
        left: 56px;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.96);
        color: #1d1d1f;
        padding: 8px 14px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.28);
        font-family: Arial, sans-serif;
        line-height: 1.3;
        font-size: 13px;
        max-width: calc(100vw - 440px);
      }}
      #folium-frio-card {{
        position: fixed;
        top: 86px;
        left: 10px;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.97);
        color: #1d1d1f;
        padding: 8px 12px;
        border-radius: 8px;
        border-left: 3.5px solid #0288D1;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.28);
        font-family: Arial, sans-serif;
        font-size: 11px;
        width: 286px;
      }}
      #folium-frio-card .frio-top-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
        margin: 6px 0 5px 0;
      }}
      #folium-frio-card .frio-toggle-lbl {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 11px;
        font-weight: 600;
        color: #01579B;
        cursor: pointer;
      }}
      #folium-frio-card .frio-btn-all {{
        border: 1px solid #0288D1;
        background: #E1F5FE;
        color: #01579B;
        border-radius: 5px;
        padding: 2px 7px;
        font-size: 10.5px;
        font-weight: 700;
        cursor: pointer;
      }}
      #folium-frio-card .frio-btn-all:hover {{
        background: #B3E5FC;
      }}
      #folium-frio-card .frio-btn-list {{
        display: flex;
        flex-direction: column;
        gap: 4px;
      }}
      #folium-frio-card .frio-btn {{
        text-align: left;
        border: 1px solid #CFD8DC;
        background: #F8FAFC;
        color: #1E293B;
        border-radius: 5px;
        padding: 4px 8px;
        font-size: 10.8px;
        cursor: pointer;
        transition: background 0.15s ease, border-color 0.15s ease;
      }}
      #folium-frio-card .frio-btn:hover {{
        background: #E0F2FE;
        border-color: #0288D1;
      }}
      #folium-frio-card .frio-btn-opt {{
        background: #E1F5FE;
        border-color: #0288D1;
        color: #01579B;
        font-weight: 700;
      }}
      #folium-frio-card .frio-cand-row {{
        margin-top: 7px;
        padding-top: 6px;
        border-top: 1px solid #E2E8F0;
      }}
      #folium-frio-card .frio-cand-lbl {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        font-weight: 600;
        color: #5B21B6;
        cursor: pointer;
        user-select: none;
      }}
      /* ── Paso 1: Ficha Rápida Contextual FLP (sin tapar el mapa) ── */
      #folium-frio-quickcard {{
        display: none;
        position: fixed;
        top: 86px;
        left: 306px;
        z-index: 1005;
        width: 344px;
        max-width: calc(100vw - 320px);
        background: #141418;
        color: #f5f5f7;
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.45);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 11.5px;
        line-height: 1.45;
      }}
      #folium-frio-quickcard.open {{
        display: block;
      }}
      .fq-head {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 8px;
      }}
      .fq-badge {{
        display: inline-block;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 2px 7px;
        border-radius: 999px;
        background: rgba(16, 185, 129, 0.16);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        margin-bottom: 3px;
      }}
      .fq-badge.cand {{
        background: rgba(128, 82, 255, 0.16);
        color: #c4b5fd;
        border-color: rgba(128, 82, 255, 0.4);
      }}
      .fq-title {{
        font-size: 13.5px;
        font-weight: 700;
        color: #f5f5f7;
        margin: 0;
      }}
      .fq-close {{
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #a1a1aa;
        border-radius: 6px;
        width: 24px;
        height: 24px;
        font-size: 14px;
        line-height: 1;
        cursor: pointer;
        flex-shrink: 0;
      }}
      .fq-close:hover {{
        background: rgba(255, 255, 255, 0.14);
        color: #fff;
      }}
      .fq-pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-bottom: 9px;
      }}
      .fq-pill {{
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: #1d1d22;
        color: #a1a1aa;
        border-radius: 6px;
        padding: 3px 7px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 10.5px;
        font-weight: 600;
        cursor: pointer;
      }}
      .fq-pill:hover {{
        color: #fff;
        border-color: rgba(128, 82, 255, 0.6);
      }}
      .fq-pill.active {{
        background: rgba(128, 82, 255, 0.24);
        color: #fff;
        border-color: #8052ff;
      }}
      .fq-kpis {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        margin-bottom: 9px;
      }}
      .fq-kpi {{
        background: #1b1b20;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 6px 8px;
      }}
      .fq-kpi-lbl {{
        font-size: 9.5px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #9ca3af;
      }}
      .fq-kpi-val {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-variant-numeric: tabular-nums;
        font-size: 13px;
        font-weight: 700;
        color: #f5f5f7;
        margin-top: 2px;
      }}
      .fq-kpi-sub {{
        font-size: 9.8px;
        color: #a1a1aa;
        margin-top: 1px;
      }}
      .fq-bullets {{
        background: #101014;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 7px 9px;
        margin-bottom: 9px;
        font-size: 10.8px;
        color: #d4d4d8;
      }}
      .fq-bullets div + div {{
        margin-top: 4px;
      }}
      .fq-bullets strong {{
        color: #f5f5f7;
      }}
      .fq-cta {{
        width: 100%;
        border: 1px solid rgba(167, 139, 250, 0.45);
        background: #7C3AED;
        color: #ffffff;
        border-radius: 8px;
        padding: 8px 10px;
        font-size: 11.5px;
        font-weight: 700;
        cursor: pointer;
        transition: background 0.15s ease;
      }}
      .fq-cta:hover {{
        background: #6D28D9;
      }}
      #folium-itin-card {{
        position: fixed;
        top: 235px;
        left: 10px;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.96);
        color: #1d1d1f;
        padding: 8px 12px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.28);
        font-family: Arial, sans-serif;
        font-size: 11px;
        max-height: calc(100vh - 280px);
        overflow-y: auto;
      }}
      #folium-legend-card {{
        position: fixed;
        bottom: 24px;
        right: 10px;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.96);
        color: #1d1d1f;
        padding: 8px 12px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.28);
        font-family: Arial, sans-serif;
        font-size: 11px;
        line-height: 17px;
        max-height: 44vh;
        overflow-y: auto;
      }}
      #folium-legend-card summary,
      #folium-frio-card summary,
      #folium-itin-card summary {{
        cursor: pointer;
        user-select: none;
        outline: none;
      }}
      .leaflet-top.leaflet-right .leaflet-control-layers-expanded {{
        max-height: calc(100vh - 340px);
        overflow-y: auto;
        overflow-x: hidden;
        box-sizing: border-box;
        background: rgba(255, 255, 255, 0.96);
      }}
    </style>
    <div id="folium-title-card">
      <b>🐟 Cadena de Frío — Pesca Artesanal</b><br>
      <span style="font-size:11.5px; color:#555;">
        Puerto Gaitán, Meta, Colombia
      </span>
    </div>
    <div id="folium-frio-card">
      <details open>
        <summary><b>❄️ Centros de Frío ({len(frio_markers_meta)})</b></summary>
        <div class="frio-top-row">
          <label class="frio-toggle-lbl">
            <input type="checkbox" id="chk-centros-frio" checked onchange="window.toggleCentrosFrio(this.checked)">
            Mostrar en mapa
          </label>
          <button type="button" class="frio-btn-all" onclick="window.verTodosCentrosFrio()">Ver todos</button>
        </div>
        <div class="frio-btn-list">
          {botones_frio_html}
        </div>
        <div class="frio-cand-row">
          <label class="frio-cand-lbl" title="Mostrar u ocultar todos los nodos candidatos a centro de frío (F01, F02, F03, A01, A02)">
            <input type="checkbox" id="chk-candidatos-frio" onchange="window.toggleCandidatosFrio(this.checked)">
            🎯 Nodos candidatos a frío ({len(candidatos_meta)})
          </label>
        </div>
      </details>
    </div>
    <div id="folium-frio-quickcard" aria-live="polite"></div>
    <script>
      (function () {{
        var FRIO_DATA = {frio_json};
        var CAND_DATA = {cand_json};
        var QUICK_FLP = {quick_flp_json};
        var quickActivo = 'F02';

        function getMapAndLayer() {{
          var m = window['{map_js_var}'];
          var fg = window['{frio_js_var}'];
          var cg = window['{cand_js_var}'];
          return {{ map: m, layer: fg, candLayer: cg }};
        }}

        window.cerrarFichaRapidaFLP = function () {{
          var card = document.getElementById('folium-frio-quickcard');
          if (card) card.classList.remove('open');
        }};

        window.abrirCalculoFLP = function (candId) {{
          var cid = candId || quickActivo || 'F02';
          try {{
            if (window.parent && window.parent !== window) {{
              window.parent.postMessage({{ type: 'agyAbrirCalculoFLP', candidato: cid }}, '*');
              return;
            }}
          }} catch (e) {{}}
          if (typeof window.agyCambiarSlide === 'function') {{
            window.agyCambiarSlide(6, false);
            if (typeof window.agySeleccionarCandidatoFLP === 'function') {{
              window.agySeleccionarCandidatoFLP(cid);
            }}
          }}
        }};

        window.mostrarFichaRapidaFLP = function (candId) {{
          var card = document.getElementById('folium-frio-quickcard');
          var d = QUICK_FLP[candId] || QUICK_FLP['F02'];
          if (!card || !d) return;
          quickActivo = d.id;
          var orden = ['F01', 'F02', 'F03', 'A01', 'A02'];
          var pillsHtml = orden.filter(function (k) {{ return QUICK_FLP[k]; }}).map(function (k) {{
            var lbl = k === 'F02' ? '★ F02' : k;
            var cls = 'fq-pill' + (k === d.id ? ' active' : '');
            return '<button type="button" class="' + cls + '" onclick="window.seleccionarCandidatoQuick(\\'' + k + '\\')">' + lbl + '</button>';
          }}).join('');
          var tMaxTxt = d.tiempo_max_h !== null ? (d.tiempo_max_h.toFixed(2) + ' h') : '∞';
          var tColor = (d.tiempo_max_h !== null && d.tiempo_max_h <= 4.0) ? '#34d399' : '#f87171';
          var cobColor = d.cobertura_pct >= 99.9 ? '#34d399' : '#fbbf24';
          var costoLabel = d.tipo === 'F'
            ? ('$' + (d.costo_eval_esc / 1e6).toFixed(0) + 'M / mes')
            : ('$' + (d.costo_fijo_csv / 1e6).toFixed(2) + 'M / mes');
          var costoSub = d.tipo === 'F'
            ? ('CSV: $' + (d.costo_fijo_csv / 1e6).toFixed(1) + 'M · Eval.: $5M')
            : ('Costo fijo en nodos_centros.csv');
          card.innerHTML =
            '<div class="fq-head">' +
              '<div>' +
                '<span class="fq-badge' + (d.optimo ? '' : ' cand') + '">' +
                  (d.optimo ? '★ SELECCIÓN ÓPTIMA FLP' : 'CANDIDATO EVALUADO · TIPO ' + d.tipo) +
                '</span>' +
                '<h4 class="fq-title">' + d.id + ' · ' + d.nombre + '</h4>' +
              '</div>' +
              '<button type="button" class="fq-close" onclick="window.cerrarFichaRapidaFLP()" title="Cerrar ficha rápida">×</button>' +
            '</div>' +
            '<div class="fq-pills">' + pillsHtml + '</div>' +
            '<div class="fq-kpis">' +
              '<div class="fq-kpi" title="Asociaciones alcanzables en t ≤ 4.0 h mediante Dijkstra (criterio=tiempo_h, evitar_restringidos=True). Cubiertos: ' + d.cubiertos_ids.join(', ') + '">' +
                '<div class="fq-kpi-lbl">Cobertura FAO (≤4h)</div>' +
                '<div class="fq-kpi-val" style="color:' + cobColor + ';">' + d.cobertura_n + '/' + d.cobertura_total + ' (' + d.cobertura_pct.toFixed(1) + '%)</div>' +
                '<div class="fq-kpi-sub">' + d.cubiertos_ids.join(', ') + '</div>' +
              '</div>' +
              '<div class="fq-kpi" title="Tiempo máximo de llegada por Dijkstra (tiempo_h) entre las asociaciones cubiertas (límite FAO = 4.0 h)">' +
                '<div class="fq-kpi-lbl">Tiempo Máx. Acceso</div>' +
                '<div class="fq-kpi-val" style="color:' + tColor + ';">' + tMaxTxt + '</div>' +
                '<div class="fq-kpi-sub">Límite FAO: 4.00 h</div>' +
              '</div>' +
              '<div class="fq-kpi" title="capacidad_frio_kg en nodos_centros.csv frente a oferta total censada de 265.75 kg/día">' +
                '<div class="fq-kpi-lbl">Capacidad / Uso</div>' +
                '<div class="fq-kpi-val">' + d.capacidad_kg.toLocaleString() + ' kg</div>' +
                '<div class="fq-kpi-sub">Uso: ' + d.uso_pct.toFixed(1) + '% (' + d.oferta_kg + ' kg/d)</div>' +
              '</div>' +
              '<div class="fq-kpi" title="costo_fijo en nodos_centros.csv (usado por facility_location.py) vs COSTO_FIJO_CENTRO_FRIO = $5M en EvaluadorEscenarios">' +
                '<div class="fq-kpi-lbl">Costo Fijo Mensual</div>' +
                '<div class="fq-kpi-val">' + costoLabel + '</div>' +
                '<div class="fq-kpi-sub">' + costoSub + '</div>' +
              '</div>' +
            '</div>' +
            '<div class="fq-bullets">' +
              '<div><strong>Origen:</strong> ' + d.origen + '</div>' +
              '<div><strong>Filtro:</strong> <code>' + d.filtro + '</code></div>' +
              '<div><strong>Clave:</strong> ' + d.clave + '</div>' +
            '</div>' +
            '<button type="button" class="fq-cta" onclick="window.abrirCalculoFLP(\\'' + d.id + '\\')">' +
              'Ver cálculo, variables y fórmulas completas →' +
            '</button>';
          card.classList.add('open');
          ajustarPanelesFolium();
        }};

        window.seleccionarCandidatoQuick = function (candId) {{
          var ref = getMapAndLayer();
          var d = QUICK_FLP[candId];
          window.mostrarFichaRapidaFLP(candId);
          if (ref.map && d) {{
            var z = candId === 'F03' ? 13 : 14;
            ref.map.flyTo([d.lat, d.lon], z, {{ duration: 0.55 }});
          }}
        }};

        window.toggleCentrosFrio = function (mostrar) {{
          var ref = getMapAndLayer();
          if (!ref.map || !ref.layer) return;
          if (mostrar) {{
            if (!ref.map.hasLayer(ref.layer)) ref.map.addLayer(ref.layer);
          }} else {{
            if (ref.map.hasLayer(ref.layer)) ref.map.removeLayer(ref.layer);
          }}
        }};

        window.toggleCandidatosFrio = function (mostrar) {{
          var ref = getMapAndLayer();
          if (!ref.map || !ref.candLayer) return;
          if (mostrar) {{
            if (!ref.map.hasLayer(ref.candLayer)) ref.map.addLayer(ref.candLayer);
            if (CAND_DATA.length) {{
              var pts = CAND_DATA.map(function (c) {{ return [c.lat, c.lon]; }});
              ref.map.fitBounds(pts, {{ padding: [70, 70], maxZoom: 12 }});
            }}
            window.mostrarFichaRapidaFLP('F02');
          }} else {{
            if (ref.map.hasLayer(ref.candLayer)) ref.map.removeLayer(ref.candLayer);
            window.cerrarFichaRapidaFLP();
          }}
        }};

        window.verTodosCentrosFrio = function () {{
          var ref = getMapAndLayer();
          if (!ref.map || !FRIO_DATA.length) return;
          var chk = document.getElementById('chk-centros-frio');
          if (chk) chk.checked = true;
          window.toggleCentrosFrio(true);
          var pts = FRIO_DATA.map(function (f) {{ return [f.lat, f.lon]; }});
          ref.map.fitBounds(pts, {{ padding: [70, 70], maxZoom: 12 }});
          window.mostrarFichaRapidaFLP('F02');
        }};

        window.enfocarCentroFrio = function (idFrio) {{
          var ref = getMapAndLayer();
          if (!ref.map) return;
          var item = FRIO_DATA.find(function (f) {{ return f.id === idFrio; }});
          if (!item) return;
          var chk = document.getElementById('chk-centros-frio');
          if (chk) chk.checked = true;
          window.toggleCentrosFrio(true);
          var z = item.id === 'F03' ? 13 : 14;
          ref.map.flyTo([item.lat, item.lon], z, {{ duration: 0.7 }});
          var mk = window[item.marker_var];
          if (mk && typeof mk.openPopup === 'function') {{
            setTimeout(function () {{ mk.openPopup(); }}, 350);
          }}
          window.mostrarFichaRapidaFLP(idFrio);
        }};

        function sincronizarEventosMapa() {{
          var ref = getMapAndLayer();
          if (!ref.map) return;
          ref.map.on('overlayadd', function (e) {{
            if (ref.layer && e.layer === ref.layer) {{
              var chk = document.getElementById('chk-centros-frio');
              if (chk) chk.checked = true;
            }}
            if (ref.candLayer && e.layer === ref.candLayer) {{
              var chkC = document.getElementById('chk-candidatos-frio');
              if (chkC) chkC.checked = true;
            }}
          }});
          ref.map.on('overlayremove', function (e) {{
            if (ref.layer && e.layer === ref.layer) {{
              var chk = document.getElementById('chk-centros-frio');
              if (chk) chk.checked = false;
            }}
            if (ref.candLayer && e.layer === ref.candLayer) {{
              var chkC = document.getElementById('chk-candidatos-frio');
              if (chkC) chkC.checked = false;
            }}
          }});
        }}

        function ajustarPanelesFolium() {{
          if (window.innerHeight < 120) return;
          var title = document.getElementById('folium-title-card');
          var frio = document.getElementById('folium-frio-card');
          var quick = document.getElementById('folium-frio-quickcard');
          var itin = document.getElementById('folium-itin-card');
          var legend = document.getElementById('folium-legend-card');
          var zoom = document.querySelector('.leaflet-control-zoom');
          var layers = document.querySelector('.leaflet-control-layers');

          // 1. Apilar limpiamente panel izquierdo: Título/Zoom -> Centros de Frío -> Itinerarios
          var cursorY = 0;
          if (title) cursorY = Math.max(cursorY, title.getBoundingClientRect().bottom);
          if (zoom) cursorY = Math.max(cursorY, zoom.getBoundingClientRect().bottom);

          var frioTop = 86;
          if (frio && cursorY > 20) {{
            frioTop = Math.ceil(cursorY + 10);
            frio.style.setProperty('top', frioTop + 'px', 'important');
            cursorY = frio.getBoundingClientRect().bottom;
          }}

          // Alinear ficha rápida FLP a la derecha de #folium-frio-card sin solapar capas
          if (quick) {{
            var frioRight = frio ? Math.ceil(frio.getBoundingClientRect().right + 10) : 306;
            quick.style.setProperty('top', frioTop + 'px', 'important');
            quick.style.setProperty('left', frioRight + 'px', 'important');
          }}

          if (itin && cursorY > 20) {{
            var itinTop = Math.ceil(cursorY + 10);
            itin.style.setProperty('top', itinTop + 'px', 'important');
            itin.style.maxHeight = Math.max(100, window.innerHeight - itinTop - 42) + 'px';
          }}

          // 2. Evitar superposición derecha: control de capas siempre 14px arriba de la leyenda
          if (layers && legend) {{
            var layersRect = layers.getBoundingClientRect();
            var legendRect = legend.getBoundingClientRect();
            if (legendRect.top > 0 && layersRect.height > 0) {{
              var disp = Math.floor(legendRect.top - layersRect.top - 14);
              if (disp > 90) {{
                layers.style.maxHeight = disp + 'px';
                layers.style.overflowY = 'auto';
              }}
            }}
          }}
        }}

        window.addEventListener('load', function () {{
          sincronizarEventosMapa();
          ajustarPanelesFolium();
          setTimeout(ajustarPanelesFolium, 200);
          setTimeout(ajustarPanelesFolium, 700);
        }});
        window.addEventListener('resize', ajustarPanelesFolium);
        document.addEventListener('toggle', ajustarPanelesFolium, true);
        if (typeof ResizeObserver !== 'undefined') {{
          var ro = new ResizeObserver(function () {{
            ajustarPanelesFolium();
          }});
          ro.observe(document.documentElement);
        }}
      }})();
    </script>
    """
    mapa.get_root().html.add_child(folium.Element(titulo_html))

    # ── Control de capas ─────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(mapa)

    # ── Presentación interactiva por diapositivas (Mapa + Grafos Vial, Ríos y Mixto) ──
    try:
        from .graph_slides import generar_html_slides, generar_presentacion_html_autonoma
        slides_html = generar_html_slides(grafo, geo_cache, itinerarios)
        mapa.get_root().html.add_child(folium.Element(slides_html))
        generar_presentacion_html_autonoma(grafo, geo_cache, itinerarios)
    except Exception as exc:
        print(f"  [!] No se pudieron inyectar diapositivas de grafos: {exc}")

    # ── Guardar ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ruta_archivo = OUTPUT_DIR / nombre_archivo
    mapa.save(str(ruta_archivo))

    print(f"  [✓] Mapa interactivo generado: {ruta_archivo}")

    if abrir_navegador:
        import webbrowser
        webbrowser.open(str(ruta_archivo))

    return str(ruta_archivo)


# ═══════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════

# Estilo por tipo de vía: (color, grosor, dash_array)
_ESTILO_VIA = {
    "fluvial":      ("#1565C0", 5, "10 8"),
    "acceso_local": ("#757575", 4, None),
    "terrestre":    ("#E65100", 4, None),
}
_NOMBRE_VIA = {
    "fluvial": "Fluvial",
    "acceso_local": "Acceso local",
    "terrestre": "Terrestre",
}


def _agregar_capa_rios(mapa: folium.Map) -> None:
    """Agrega la capa de ríos reales (siempre visible) con tooltip del nombre."""
    datos = None
    try:
        from .river_network import obtener_capa_rios_geojson
        datos = obtener_capa_rios_geojson(tolerancia=0.0008)
    except Exception as exc:  # módulo ausente o error de lectura
        print(f"  [!] Capa de ríos no disponible: {exc}")
    if not datos or not datos.get("features"):
        return

    for idx, feat in enumerate(datos.get("features", []), start=1):
        props = feat.setdefault("properties", {})
        if not props.get("nombre"):
            props["nombre"] = f"Tramo fluvial #{idx} (Red Hidrográfica Meta)"

    capa = folium.FeatureGroup(name="🌊 Red Fluvial (ríos reales)", show=True)
    folium.GeoJson(
        datos,
        style_function=lambda _f: {
            "color": "#1E88E5", "weight": 2.5, "opacity": 0.85,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["nombre"], aliases=["Río:"], labels=True, sticky=True,
        ),
    ).add_to(capa)
    capa.add_to(mapa)


def _tiene_propiedad(datos: dict, clave: str) -> bool:
    """Indica si todas las features del GeoJSON tienen la propiedad dada."""
    feats = datos.get("features", [])
    return bool(feats) and all(
        clave in (f.get("properties") or {}) for f in feats
    )


def _locs_tramo(grafo: GrafoLogistico, geo_cache: dict, o: str, d: str) -> list:
    """Coordenadas [[lat, lon], ...] de un tramo (geometría real o recta)."""
    geo = geo_cache.get(f"{o}-{d}")
    if geo and len(geo) > 1:
        return [[c[1], c[0]] for c in geo]
    n_o, n_d = grafo.obtener_nodo(o), grafo.obtener_nodo(d)
    if n_o and n_d:
        return [[n_o.latitud, n_o.longitud], [n_d.latitud, n_d.longitud]]
    return []


def _paradas(it) -> list:
    """Lista ordenada de IDs de parada de un itinerario."""
    paradas = [it.origen_id]
    for t in it.tramos:
        paradas.append(t.destino)
    return paradas


def _resumen_acopios(itinerarios: list) -> Dict[str, dict]:
    """Por acopio: puntos de pesca que recibe y kg/día totales."""
    res: Dict[str, dict] = {}
    for it in itinerarios:
        if not getattr(it, "acopio_id", None):
            continue
        r = res.setdefault(it.acopio_id, {"origenes": set(), "kg": 0.0})
        r["origenes"].add(it.origen_id)
        r["kg"] += float(getattr(it, "kg_inicial", 0.0) or 0.0)
    return res


def _tramo_popup(grafo: GrafoLogistico, t) -> str:
    """HTML del popup de un tramo."""
    n_o, n_d = grafo.obtener_nodo(t.origen), grafo.obtener_nodo(t.destino)
    no = n_o.nombre if n_o else t.origen
    nd = n_d.nombre if n_d else t.destino
    efecto = getattr(t, "efecto", "") or "—"
    return (
        f"<b>{no} → {nd}</b><br>"
        f"🛣️ Vía: {_NOMBRE_VIA.get(t.tipo_via, t.tipo_via)}<br>"
        f"📏 {t.distancia_km:.1f} km<br>⏱️ {t.tiempo_h:.2f} h<br>"
        f"✨ Efecto: {efecto}"
    )


def _icono_numerado(num: int, color: str) -> folium.DivIcon:
    """Marca circular numerada para una parada."""
    html = (
        f'<div style="background:{color};color:white;border:2px solid white;'
        f'border-radius:50%;width:22px;height:22px;line-height:18px;'
        f'text-align:center;font:bold 12px Arial;'
        f'box-shadow:0 0 3px rgba(0,0,0,.6);">{num}</div>'
    )
    return folium.DivIcon(html=html, icon_size=(22, 22), icon_anchor=(11, 11))


def _agregar_itinerarios(mapa, grafo, itinerarios, geo_cache) -> None:
    """Una capa por itinerario + capa resumen 'Flujo de acopio'."""
    flujo = folium.FeatureGroup(name="📦 Flujo de acopio (P→A / P→M→A)", show=False)

    for i, it in enumerate(itinerarios):
        capa = folium.FeatureGroup(
            name=(f"🐟 Recorrido {it.origen_id} → Acopio → Frío → Destino"
                  f" ({it.acopio_id or '—'} / {it.centro_frio_id or '—'}"
                  f" / {it.destino_id or '—'})"),
            show=(i == 0),
        )
        llegado_a_acopio = False
        for t in it.tramos:
            locs = _locs_tramo(grafo, geo_cache, t.origen, t.destino)
            if len(locs) < 2:
                continue
            color, peso, dash = _ESTILO_VIA.get(t.tipo_via, ("#E65100", 4, None))
            folium.PolyLine(
                locs, color=color, weight=peso, opacity=0.9, dash_array=dash,
                popup=folium.Popup(_tramo_popup(grafo, t), max_width=300),
                tooltip=f"{t.origen}→{t.destino}: {t.distancia_km:.1f} km",
            ).add_to(capa)
            if not llegado_a_acopio:  # tramos hasta alcanzar el acopio
                folium.PolyLine(
                    locs, color=color, weight=peso + 1, opacity=0.65,
                    dash_array=dash,
                    tooltip=f"{it.origen_id} → {it.acopio_id}: aporta "
                            f"{getattr(it, 'kg_inicial', 0):.0f} kg/día",
                ).add_to(flujo)
                n_d = grafo.obtener_nodo(t.destino)
                if n_d and n_d.tipo == TipoNodo.ACOPIO:
                    llegado_a_acopio = True

        for k, nid in enumerate(_paradas(it), start=1):
            n = grafo.obtener_nodo(nid)
            if not n:
                continue
            color = COLORES_FELT.get(n.tipo, "#757575")
            folium.Marker(
                [n.latitud, n.longitud],
                icon=_icono_numerado(k, color),
                tooltip=f"Parada {k}: {n.nombre} ({n.id})",
            ).add_to(capa)
        capa.add_to(mapa)

    flujo.add_to(mapa)


def _agregar_nodos_especiales(mapa, grafo, itinerarios) -> None:
    """Capas de muelles (M), zonas de deterioro (D) y hábitats (H)."""
    evitados: Dict[str, list] = {}
    for it in itinerarios:
        for h in (getattr(it, "nodos_restringidos_evitados", None) or []):
            evitados.setdefault(h, []).append(it.origen_id)

    config = [
        (TipoNodo.MUELLE, "⚓ Muelles (transbordo fluvial→terrestre)", "anchor", "purple"),
        (TipoNodo.DETERIORO, "⚠️ Zonas de deterioro (×1.5)", "exclamation-triangle", "orange"),
        (TipoNodo.HABITAT, "🌿 Hábitats protegidos (restringidos)", "leaf", "green"),
    ]
    for tipo, titulo, icono, color in config:
        nodos = grafo.nodos_por_tipo(tipo)
        if not nodos:
            continue
        capa = folium.FeatureGroup(name=titulo, show=True)
        for n in nodos:
            if tipo == TipoNodo.MUELLE:
                efecto = ("Transbordo fluvial → terrestre: añade tiempo de "
                          "espera y la carga pierde frío durante el cambio.")
            elif tipo == TipoNodo.DETERIORO:
                efecto = ("Zona de deterioro: el tramo entrante multiplica "
                          "el deterioro del pescado ×1.5 (calor, sin frío).")
            else:
                ev = sorted(set(evitados.get(n.id, [])))
                efecto = ("Hábitat protegido: tránsito restringido, las rutas "
                          "lo evitan como nodo intermedio.<br>"
                          + ("🚫 Evitado por itinerarios de: " + ", ".join(ev)
                             if ev else "🚫 Ningún itinerario necesitó desviarse."))
            popup = f"<b>{n.nombre}</b> ({n.id})<br>{efecto}"
            if tipo == TipoNodo.DETERIORO:
                folium.Circle(
                    [n.latitud, n.longitud], radius=1500, color="#FB8C00",
                    weight=1, fill=True, fill_color="#FB8C00", fill_opacity=0.25,
                    tooltip="Zona de deterioro: deterioro ×1.5",
                ).add_to(capa)
            elif tipo == TipoNodo.HABITAT:
                folium.Circle(
                    [n.latitud, n.longitud], radius=2000, color="#D32F2F",
                    weight=2, dash_array="6 6", fill=True,
                    fill_color="#43A047", fill_opacity=0.25,
                    tooltip="Hábitat protegido — tránsito restringido; rutas lo evitan",
                ).add_to(capa)
            folium.Marker(
                [n.latitud, n.longitud],
                icon=folium.Icon(color=color, icon=icono, prefix="fa"),
                popup=folium.Popup(popup, max_width=300),
                tooltip=f"{n.nombre} ({TipoNodo.descripcion(tipo)})",
            ).add_to(capa)
        capa.add_to(mapa)


def _agregar_leyenda(mapa: folium.Map) -> None:
    """Leyenda: tipos de nodo, tipos de vía y círculos D/H."""
    nodos = [
        ("#2196F3", "Punto de Pesca (P)"), ("#FF9800", "Centro de Acopio (A)"),
        ("#00BCD4", "Centro de Frío (F)"), ("#E91E63", "Venta (V)"),
        ("#9C27B0", "Muelle (M) ⚓"), ("#F57C00", "Deterioro (D) ⚠"),
        ("#4CAF50", "Hábitat protegido (H) 🌿"),
    ]
    vias = [
        ("#1565C0", "Fluvial (punteada)", "10 8"), ("#757575", "Acceso local", None),
        ("#E65100", "Terrestre", None), ("#1E88E5", "Río real", None),
    ]
    h = ('<div id="folium-legend-card">'
         '<details open><summary><b>Leyenda</b></summary><div style="margin-top:4px;">')
    for c, l in nodos:
        h += (f'<span style="display:inline-block;width:11px;height:11px;'
              f'background:{c};border-radius:50%;margin-right:6px;'
              f'vertical-align:middle;"></span>{l}<br>')
    h += '<b>Vías</b><br>'
    for c, l, d in vias:
        borde = "dashed" if d else "solid"
        h += (f'<span style="display:inline-block;width:20px;height:0;'
              f'border-top:3px {borde} {c};margin-right:6px;'
              f'vertical-align:middle;"></span>{l}<br>')
    h += ('<b>Zonas</b><br>'
          '<span style="display:inline-block;width:11px;height:11px;'
          'background:rgba(251,140,0,.35);border:1px solid #FB8C00;'
          'border-radius:50%;margin-right:6px;vertical-align:middle;"></span>'
          'Deterioro ×1.5 (r=1.5 km)<br>'
          '<span style="display:inline-block;width:11px;height:11px;'
          'background:rgba(67,160,71,.35);border:2px dashed #D32F2F;'
          'border-radius:50%;margin-right:6px;vertical-align:middle;"></span>'
          'Hábitat restringido (r=2 km)</div></details></div>')
    mapa.get_root().html.add_child(folium.Element(h))


def _agregar_panel_itinerarios(mapa, grafo, itinerarios) -> None:
    """Panel lateral con la tabla resumen de itinerarios."""
    filas = ""
    for it in itinerarios:
        filas += (
            f"<tr><td>{it.origen_id}</td><td>{it.acopio_id or '—'}</td>"
            f"<td>{it.centro_frio_id or '—'}</td>"
            f"<td align='right'>{it.distancia_total_km:.1f}</td>"
            f"<td align='right'>{it.tiempo_total_h:.2f}</td>"
            f"<td align='right'>{it.perdida_pct:.1f}%</td></tr>"
        )
    h = (
        '<div id="folium-itin-card">'
        '<details open><summary><b>📋 Itinerarios por punto de pesca</b></summary>'
        '<table style="border-collapse:collapse;margin-top:4px;color:#1d1d1f;" cellpadding="3">'
        '<tr style="background:#eee;color:#1d1d1f;"><th>Origen</th><th>Acopio</th>'
        '<th>Centro</th><th>km</th><th>h</th><th>% pérdida</th></tr>'
        f'{filas}</table></details></div>'
    )
    mapa.get_root().html.add_child(folium.Element(h))


def _agregar_puntos_campo(mapa: folium.Map) -> None:
    """
    Carga los 49 puntos de campo desde data/geojson/puntos_campo.geojson,
    calcula su corredor de navegación en lancha sobre la red fluvial real (rios.shp)
    incluyendo condiciones de cuesta del río (aguas abajo vs. cuesta arriba contra corriente)
    y los agrega como capas activas al mapa interactivo.
    """
    try:
        from .river_network import calcular_corredor_fluvial_puntos_campo
        corredor = calcular_corredor_fluvial_puntos_campo()
    except Exception:
        corredor = {"puntos": [], "tramos": []}

    puntos = corredor.get("puntos", [])
    if not puntos:
        return

    # Configuración visual por categoría
    categorias_config = {
        "A": {"nombre": "📦 Puntos de Acopio (campo — conectado por río)",    "color": "#F39C12", "icono": "warehouse"},
        "D": {"nombre": "🌡️ Zonas de Deterioro (campo — conectado por río)", "color": "#E74C3C", "icono": "warning"},
        "H": {"nombre": "🌿 Hábitat Protegido (campo — referencia fluvial)",  "color": "#27AE60", "icono": "leaf"},
        "M": {"nombre": "⚓ Muelles (campo — conectado por río)",             "color": "#3498DB", "icono": "anchor"},
        "V": {"nombre": "🏪 Puntos de Venta (campo — conectado por río)",     "color": "#8E44AD", "icono": "shop"},
    }

    # 1. Capa de recorridos en lancha por la Red Fluvial real (rios.shp)
    capa_rutas_campo = folium.FeatureGroup(
        name="🚤 Recorridos en Lancha — Red Fluvial de Campo (49 puntos, rios.shp)",
        show=True,
    )

    for p in puntos:
        coords_ll = p.get("coords_latlon", [])
        if len(coords_ll) < 2:
            continue
        cond_cuesta = str(p.get("condicion_cuesta_rio", ""))
        if "Aguas abajo" in cond_cuesta:
            color_linea = "#0096C7"  # Azul cian intenso: a favor de la corriente
            dash_pat = "8 5"
        elif "Cuesta arriba" in cond_cuesta:
            color_linea = "#E67E22"  # Ámbar naranja: remontando contra corriente
            dash_pat = "6 6"
        else:
            color_linea = "#2980B9"
            dash_pat = "8 6"

        tramo_popup = (
            f"<div style='font-family:Segoe UI,sans-serif;font-size:12px;line-height:1.45;'>"
            f"<b>🚤 Tramo Fluvial en Lancha: {p['id']} → {p['destino_relevo_id']}</b><br>"
            f"<b>Origen:</b> {p['title']} ({p['cota_origen_msnm']} m s.n.m.)<br>"
            f"<b>Relevo destino:</b> {p['destino_relevo_nombre']} ({p['cota_destino_msnm']} m s.n.m.)<br>"
            f"<hr style='margin:4px 0'>"
            f"<b>Distancia por río real (rios.shp):</b> {p['distancia_tramo_rio_km']} km "
            f"(sinuosidad {p['indice_sinuosidad']}×)<br>"
            f"<b>Distancia total a Pto. Gaitán (M03):</b> {p['distancia_total_a_m03_km']} km<br>"
            f"<b>Condición de cuesta del río:</b> {cond_cuesta}<br>"
            f"<b>Desnivel a M03:</b> {p['desnivel_a_m03_m']:+.1f} m "
            f"(pendiente {p['pendiente_m_km']} m/km)<br>"
            f"<b>Corriente fluvial:</b> {p['velocidad_corriente_kmh']:+.1f} km/h | "
            f"<b>Velocidad lancha:</b> {p['velocidad_lancha_ida_kmh']} km/h<br>"
            f"<b>Tiempo tramo:</b> {p['tiempo_tramo_h']} h ({p['tiempo_tramo_min']} min) | "
            f"<b>Total a M03:</b> {p['tiempo_total_a_m03_h']} h<br>"
            f"<b>Gasolina a M03:</b> {p['galones_gasolina']} gal (${p['costo_combustible_cop']:,.0f} COP)<br>"
            f"<b>Pérdida térmica sin hielo (30 °C):</b> <span style='color:#c0392b;font-weight:700;'>"
            f"{p['perdida_sin_hielo_pct']}%</span> vs. "
            f"<b>con hielo 1:1 (0 °C):</b> <span style='color:#27ae60;font-weight:700;'>"
            f"{p['perdida_con_hielo_1a1_pct']}%</span>"
            f"</div>"
        )
        folium.PolyLine(
            locations=coords_ll,
            color=color_linea,
            weight=3.2,
            opacity=0.82,
            dash_array=dash_pat,
            popup=folium.Popup(tramo_popup, max_width=340),
            tooltip=(
                f"🚤 {p['id']} → {p['destino_relevo_id']} ({p['distancia_tramo_rio_km']} km río | "
                f"{cond_cuesta} | {p['tiempo_tramo_h']} h)"
            ),
        ).add_to(capa_rutas_campo)

    capa_rutas_campo.add_to(mapa)

    # 2. Agrupar puntos por categoría y crear sus marcadores con popup completo
    grupos: Dict[str, list] = {}
    for p in puntos:
        cat = str(p.get("categoria", "?"))
        grupos.setdefault(cat, []).append(p)

    for cat, lista_pts in sorted(grupos.items()):
        config = categorias_config.get(cat, {
            "nombre": f"❓ {cat} (campo)",
            "color": "#757575",
            "icono": "circle",
        })

        capa = folium.FeatureGroup(name=config["nombre"], show=True)

        for p in lista_pts:
            lat, lon = float(p["latitud"]), float(p["longitud"])
            popup_html = (
                f"<div style='font-family:Segoe UI,sans-serif;font-size:12px;line-height:1.45;'>"
                f"<b>{p['title']}</b><br>"
                f"<i>{p['tipo']} · Cota cauce: {p['cota_origen_msnm']} m s.n.m.</i><br>"
                f"📍 {p['lat_dms']}, {p['lon_dms']}<br>"
                f"<hr style='margin:5px 0'>"
                f"<b>🌊 Conexión por Red Fluvial (rios.shp):</b><br>"
                f"• <b>Ruta de relevo:</b> {p['id']} ──río──► <b>{p['destino_relevo_id']}</b> ──► <b>M03 (Pto. Gaitán)</b><br>"
                f"• <b>Distancia tramo río:</b> {p['distancia_tramo_rio_km']} km (recta: {p['distancia_recta_km']} km, "
                f"meandros {p['indice_sinuosidad']}×)<br>"
                f"• <b>Distancia fluvial total a M03:</b> <b>{p['distancia_total_a_m03_km']} km</b> "
                f"(acceso al cauce: {p['distancia_al_cauce_km']} km)<br>"
                f"<hr style='margin:5px 0'>"
                f"<b>⛰️ Condición de Cuesta del Río y Lancha (40 HP):</b><br>"
                f"• <b>Régimen hidráulico:</b> {p['condicion_cuesta_rio']}<br>"
                f"• <b>Desnivel hacia M03:</b> {p['desnivel_a_m03_m']:+.1f} m "
                f"({p['cota_origen_msnm']} → {p['cota_m03_msnm']} m s.n.m., {p['pendiente_m_km']} m/km)<br>"
                f"• <b>Velocidad corriente:</b> {p['velocidad_corriente_kmh']:+.1f} km/h | "
                f"<b>Esfuerzo motor:</b> {p['esfuerzo_motor']}<br>"
                f"• <b>Velocidad lancha (ida / retorno con hielo):</b> "
                f"{p['velocidad_lancha_ida_kmh']} km/h / {p['velocidad_lancha_retorno_kmh']} km/h<br>"
                f"• <b>Tiempo navegación en lancha:</b> {p['tiempo_tramo_h']} h al relevo | "
                f"<b>{p['tiempo_total_a_m03_h']} h ({p['tiempo_total_a_m03_min']} min) total a M03</b><br>"
                f"<hr style='margin:5px 0'>"
                f"<b>❄️ Variables de Frío, Combustible e Inocuidad FAO:</b><br>"
                f"• <b>Combustible lancha a M03:</b> {p['galones_gasolina']} gal "
                f"(<b>${p['costo_combustible_cop']:,.0f} COP</b>)<br>"
                f"• <b>Hielo 1:1 requerido (lote {p['lote_referencia_kg']:.0f} kg):</b> "
                f"{p['hielo_requerido_1a1_kg']:.0f} kg hielo en escarcha<br>"
                f"• <b>Merma a 30 °C (sin hielo):</b> "
                f"<span style='color:#c0392b;font-weight:700;'>{p['perdida_sin_hielo_pct']}%</span> "
                f"vs. <b>con hielo 1:1 (0 °C):</b> "
                f"<span style='color:#27ae60;font-weight:700;'>{p['perdida_con_hielo_1a1_pct']}%</span><br>"
                f"• <b>Diagnóstico FAO (4.0 h):</b> <b>{p['estado_fao_sin_hielo']}</b>"
                f"</div>"
            )

            folium.CircleMarker(
                location=[lat, lon],
                radius=7,
                color="#111111",
                fill=True,
                fill_color=config["color"],
                fill_opacity=0.9,
                weight=2,
                popup=folium.Popup(popup_html, max_width=360),
                tooltip=(
                    f"{p['title']} | Río a {p['destino_relevo_id']}: {p['distancia_tramo_rio_km']} km "
                    f"(Total a M03: {p['distancia_total_a_m03_km']} km, {p['tiempo_total_a_m03_h']} h, "
                    f"{p['condicion_cuesta_rio']})"
                ),
            ).add_to(capa)

        capa.add_to(mapa)

    print(f"  [✓] {len(puntos)} puntos de campo cargados y conectados por red fluvial "
          f"({', '.join(f'{k}:{len(v)}' for k, v in sorted(grupos.items()))})")

