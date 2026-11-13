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
from ..utils.data_loader import OUTPUT_DIR, SHP_DIR
from ..utils.constants import TipoNodo


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

    # ── Capa de ríos (si existe shapefile) ───────────────────
    _agregar_capa_rios(mapa)

    # ── Capa de aristas (todas las conexiones) ───────────────
    capa_aristas = folium.FeatureGroup(name="🔗 Conexiones del Grafo")
    for arista in grafo.aristas:
        n_o = grafo.obtener_nodo(arista.origen)
        n_d = grafo.obtener_nodo(arista.destino)
        if not n_o or not n_d:
            continue

        color_via = {
            "terrestre": "#9E9E9E",
            "fluvial":   "#42A5F5",
            "acceso_local": "#BDBDBD",
        }.get(arista.tipo_via, "#9E9E9E")

        popup_html = (
            f"<b>{n_o.nombre} → {n_d.nombre}</b><br>"
            f"📏 {arista.distancia_km} km<br>"
            f"⏱️ {arista.tiempo_h:.2f} h<br>"
            f"💰 ${arista.costo_cop:,.0f} COP<br>"
            f"🌡️ Riesgo térmico: {arista.riesgo_termico}<br>"
            f"🛣️ Vía: {arista.tipo_via} ({arista.estado_via})"
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
            weight=1.5,
            opacity=0.5,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{arista.distancia_km} km | {arista.tiempo_h:.2f}h",
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
    for tipo_code in [TipoNodo.PESCA, TipoNodo.ACOPIO, TipoNodo.FRIO,
                      TipoNodo.MUELLE, TipoNodo.VENTA]:
        nodos_tipo = grafo.nodos_por_tipo(tipo_code)
        if not nodos_tipo:
            continue

        desc = TipoNodo.descripcion(tipo_code)
        icono_nombre, icono_color = ICONOS_FOLIUM.get(
            tipo_code, ("circle", "gray")
        )
        capa = folium.FeatureGroup(
            name=f"{'🐟' if tipo_code == 'P' else '❄️' if tipo_code == 'F' else '📦' if tipo_code == 'A' else '🏪' if tipo_code == 'V' else '⚓'} {desc}"
        )

        for nodo in nodos_tipo:
            # Construir popup HTML rico
            popup_lines = [f"<b>{nodo.nombre}</b><br>"]
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
                    f"💰 Costo fijo: ${nodo.costo_fijo:,.0f}/mes<br>"
                )
                popup_lines.append(
                    f"⚡ Energía: {'Sí' if nodo.tiene_energia else 'No'}<br>"
                )
            elif nodo.tipo == TipoNodo.VENTA:
                demanda = nodo.metadata.get("demanda_kg_dia", 0)
                if demanda:
                    popup_lines.append(
                        f"📊 Demanda: {demanda} kg/día<br>"
                    )

            popup_lines.append(
                f"📍 {nodo.latitud:.4f}, {nodo.longitud:.4f}"
            )

            # Determinar tamaño del marcador
            if nodo.tipo == TipoNodo.PESCA:
                radio = max(6, min(nodo.produccion_kg_dia / 5, 15))
            elif nodo.tipo in (TipoNodo.FRIO, TipoNodo.ACOPIO):
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

    # ── Leyenda HTML ─────────────────────────────────────────
    _agregar_leyenda(mapa)

    # ── Título ───────────────────────────────────────────────
    titulo_html = """
    <div style="position:fixed; top:10px; left:60px; z-index:1000;
         background:white; padding:10px 18px; border-radius:8px;
         box-shadow:0 2px 6px rgba(0,0,0,0.3); font-family:Arial;">
      <b>🐟 Cadena de Frío — Pesca Artesanal</b><br>
      <span style="font-size:12px; color:#666;">
        Puerto Gaitán, Meta, Colombia
      </span>
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(titulo_html))

    # ── Control de capas ─────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(mapa)

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

def _agregar_capa_rios(mapa: folium.Map) -> None:
    """Intenta agregar la capa de ríos desde el GeoJSON exportado o shapefile."""
    # Opción 1: Usar GeoJSON generado por felt_exporter
    geojson_rios = OUTPUT_DIR / "nodos_red.geojson"

    # Opción 2: Intentar leer el shapefile directamente
    try:
        import geopandas as gpd
        shp_path = SHP_DIR / "rios.shp"
        if shp_path.exists():
            gdf = gpd.read_file(str(shp_path))
            geojson_data = json.loads(gdf.to_json())

            capa_rios = folium.FeatureGroup(name="🌊 Red Fluvial")
            folium.GeoJson(
                geojson_data,
                style_function=lambda x: {
                    "color": "#42A5F5",
                    "weight": 1.5,
                    "opacity": 0.6,
                },
                name="Ríos",
            ).add_to(capa_rios)
            capa_rios.add_to(mapa)
            return
    except (ImportError, Exception):
        pass  # geopandas no disponible o error al leer


def _agregar_leyenda(mapa: folium.Map) -> None:
    """Agrega una leyenda HTML al mapa."""
    items = [
        ("#2196F3", "Punto de Pesca"),
        ("#FF9800", "Centro de Acopio"),
        ("#00BCD4", "Centro de Frío"),
        ("#E91E63", "Mercado / Venta"),
        ("#D32F2F", "Ruta Óptima"),
        ("#9E9E9E", "Conexiones"),
    ]

    legend_html = """
    <div style="position:fixed; bottom:30px; right:10px; z-index:1000;
         background:white; padding:12px 16px; border-radius:8px;
         box-shadow:0 2px 6px rgba(0,0,0,0.3); font-family:Arial;
         font-size:12px; line-height:20px;">
      <b>Leyenda</b><br>
    """
    for color, label in items:
        legend_html += (
            f'<span style="display:inline-block; width:12px; height:12px; '
            f'background:{color}; border-radius:50%; margin-right:6px;'
            f'vertical-align:middle;"></span>{label}<br>'
        )
    legend_html += "</div>"

    mapa.get_root().html.add_child(folium.Element(legend_html))

