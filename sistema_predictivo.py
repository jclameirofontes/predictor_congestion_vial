
import pandas as pd
import numpy as np
import joblib
import os
import googlemaps
import requests
from geopy.distance import geodesic
from shapely.geometry import Point
import folium
from folium import DivIcon
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import cKDTree


API_KEY = "AIzaSyA2rbw0-kRC3bIoU3-ycgPGZ79zXN2RFvI"
gmaps = googlemaps.Client(key=API_KEY)

ruta_modelos = "modelos"
carga_media = pd.read_parquet("id_carga_media.parquet")
ruta_coordenadas_trafico = "df_coordenadas.parquet"
df_coordenadas_trafico_default = pd.read_parquet(ruta_coordenadas_trafico)

def hora_a_sin_cos(hora_str):
    h, m = map(int, hora_str.split(":"))
    angulo = 2 * np.pi * (h + m / 60) / 24
    return np.sin(angulo), np.cos(angulo)

def direccion_viento_a_sin_cos(grados):
    radianes = np.deg2rad(grados)
    return np.sin(radianes), np.cos(radianes)

def normalizar_variable(nombre, valor_original):
    rangos = {
        "TEMPERATURA (ºC)": (3.25, 22.85),
        "HUMEDAD RELATIVA (%)": (33.0, 100.0),
        "PRESION BARIOMETRICA (mb)": (933.4286, 968.0),
        "RADIACION SOLAR (W/m2)": (0.0, 705.0),
        "VELOCIDAD VIENTO (m/s)": (0.0, 4.775)
    }
    minimo, maximo = rangos[nombre]
    return (valor_original - minimo) / (maximo - minimo)

def usar_valor_o_defecto(valor, nombre_variable):
    valores_por_defecto = {
        "TEMPERATURA": 12.0,
        "HUMEDAD": 70.0,
        "PRESION": 946.0,
        "RADIACION": 150.0,
        "VELOCIDAD_VIENTO": 1.5,
        "DIR_VIENTO": 180
    }
    if valor is None or str(valor).strip().upper() in ["NA", "NAN", ""]:
        return valores_por_defecto[nombre_variable]
    return valor

def interpolar_puntos(p1, p2, distancia_minima):
    distancia_total = geodesic(p1, p2).meters
    if distancia_total <= distancia_minima:
        return [p2]
    num_puntos = int(distancia_total // distancia_minima)
    return [(p1[0] + (p2[0] - p1[0]) * (i / num_puntos), p1[1] + (p2[1] - p1[1]) * (i / num_puntos)) for i in range(1, num_puntos + 1)]

def obtener_bounding_box(coordenadas_ruta, margen=0.01):
    lats, lons = zip(*coordenadas_ruta)
    return {
        "min_lat": min(lats) - margen, "max_lat": max(lats) + margen,
        "min_lon": min(lons) - margen, "max_lon": max(lons) + margen
    }

def filtrar_puntos_medicion(df_puntos, bounding_box, max_puntos=1000):
    df_filtrado = df_puntos[
        (df_puntos["latitud"] >= bounding_box["min_lat"]) &
        (df_puntos["latitud"] <= bounding_box["max_lat"]) &
        (df_puntos["longitud"] >= bounding_box["min_lon"]) &
        (df_puntos["longitud"] <= bounding_box["max_lon"])
    ]
    return df_filtrado.head(max_puntos)

def obtener_coordenadas_ruta(origen, destino, modo="DRIVE", distancia_minima=8):
    # 👇 Autocompletar con ", Madrid" si no se especifica ciudad
    if "madrid" not in origen.lower():
        origen += ", Madrid"
    if "madrid" not in destino.lower():
        destino += ", Madrid"

    print("🌐 Solicitando coordenadas de ruta...")
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "routes.legs.steps.polyline"
    }
    body = {
        "origin": {"address": origen},
        "destination": {"address": destino},
        "travelMode": modo
    }
    response = requests.post(url, headers=headers, json=body)
    print("🌐 Estado de la respuesta:", response.status_code)

    if response.status_code != 200:
        print("❌ Error al obtener ruta:", response.text)
        return f"Error en la API: {response.status_code}, {response.text}"

    data = response.json()
    coordenadas_ruta = []

    if "routes" in data and data["routes"]:
        for leg in data["routes"][0]["legs"]:
            for step in leg["steps"]:
                polyline_codificada = step["polyline"]["encodedPolyline"]
                puntos = googlemaps.convert.decode_polyline(polyline_codificada)
                puntos = [(float(p["lat"]), float(p["lng"])) for p in puntos]
                if coordenadas_ruta:
                    for i in range(len(puntos) - 1):
                        coordenadas_ruta.append(puntos[i])
                        coordenadas_ruta.extend(interpolar_puntos(puntos[i], puntos[i + 1], distancia_minima))
                else:
                    coordenadas_ruta.extend(puntos)

    if not coordenadas_ruta:
        raise ValueError("❌ No se pudo obtener una ruta válida. ¿Las direcciones son correctas?")

    print("✅ Coordenadas de ruta obtenidas:", len(coordenadas_ruta))
    return coordenadas_ruta



def encontrar_puntos_de_medicion(coordenadas_ruta, df_puntos, radio_metros=6):
    # Convertir grados a metros: 1 grado lat/lon ≈ 111_139 metros
    radio_grados = radio_metros / 111_139

    puntos_array = df_puntos[["latitud", "longitud"]].to_numpy()
    tree = cKDTree(puntos_array)

    puntos_cercanos = {}
    for coord in coordenadas_ruta:
        idxs = tree.query_ball_point(coord, r=radio_grados)
        for idx in idxs:
            row = df_puntos.iloc[idx]
            punto_id = row["id"]
            if punto_id not in puntos_cercanos:
                puntos_cercanos[punto_id] = {
                    "nombre": row["nombre"],
                    "latitud": row["latitud"],
                    "longitud": row["longitud"]
                }

    print("📡 Puntos cercanos encontrados:", len(puntos_cercanos))
    return puntos_cercanos


def predecir_para_puntos(puntos_cercanos, datos_contexto):
    predicciones = {}
    for id_punto, info in puntos_cercanos.items():
        fila = carga_media[carga_media["id"] == id_punto]
        if fila.empty:
            continue
        id_target = fila["id_carga_media"].values[0]
        if id_target <= 0.09155:
            grupo = 0
        elif id_target <= 0.15876:
            grupo = 1
        elif id_target <= 0.23443:
            grupo = 2
        else:
            grupo = 3
        modelo_path = os.path.join(ruta_modelos, f"modelo_g{grupo}.pkl")
        print(f"📦 Cargando modelo grupo {grupo} para punto {id_punto}...")
        modelo = joblib.load(modelo_path)
        entrada = pd.DataFrame([{**datos_contexto, "id_target": id_target}])
        pred = modelo.predict(entrada)[0]
        puntos_cercanos[id_punto]["prediccion"] = pred
        puntos_cercanos[id_punto]["grupo"] = grupo
        predicciones[id_punto] = pred
    print("✅ Predicciones realizadas para:", list(predicciones.keys()))
    return puntos_cercanos

def visualizar_ruta(coordenadas_ruta, puntos_medicion):
    if not coordenadas_ruta or isinstance(coordenadas_ruta, str):
        print("No se puede visualizar la ruta: No hay datos válidos.")
        return

    puntos_medicion = {
        id_punto: datos for id_punto, datos in puntos_medicion.items()
        if datos.get("prediccion") is not None
    }

    mapa = folium.Map(location=coordenadas_ruta[0], zoom_start=14)

    def color_por_carga(carga):
        if carga <= 0.2:
            return "green"
        elif carga <= 0.3:
            return "gold"
        elif carga <= 0.5:
            return "orange"
        else:
            return "red"

    puntos_con_carga = [
        (Point(info["longitud"], info["latitud"]), info.get("prediccion", 0.0))
        for info in puntos_medicion.values()
    ]

    distancia_maxima_metros = 1000
    for i in range(len(coordenadas_ruta) - 1):
        p1 = coordenadas_ruta[i]
        p2 = coordenadas_ruta[i + 1]
        mid_point = Point((p1[1] + p2[1]) / 2, (p1[0] + p2[0]) / 2)

        distancia_min = float("inf")
        carga_asociada = None

        for punto, carga in puntos_con_carga:
            dist_metros = mid_point.distance(punto) * 111139
            if dist_metros < distancia_min:
                distancia_min = dist_metros
                carga_asociada = carga

        color = "gray" if distancia_min > distancia_maxima_metros else color_por_carga(carga_asociada)
        folium.PolyLine([p1, p2], color=color, weight=5, opacity=0.8).add_to(mapa)

    for id_punto, datos in puntos_medicion.items():
        pred = datos.get("prediccion")
        grupo = datos.get("grupo", "?")
        if pred is not None:
            color = color_por_carga(pred)
            popup_texto = f"{datos['nombre']}<br>(ID: {id_punto})<br><b>Carga estimada:</b> {pred:.3f}<br><b>Grupo:</b> {grupo}"

            html_icon = (
                f'<div style="background-color:{color};border-radius:50%;color:white;font-size:10pt;'
                f'font-weight:bold;text-align:center;width:24px;height:24px;line-height:24px;">{grupo}</div>'
            )

            folium.Marker(
                location=(datos["latitud"], datos["longitud"]),
                popup=folium.Popup(popup_texto, max_width=300),
                icon=folium.DivIcon(html=html_icon)
            ).add_to(mapa)

    folium.Marker(coordenadas_ruta[0], popup="Inicio", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(mapa)
    folium.Marker(coordenadas_ruta[-1], popup="Destino", icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")).add_to(mapa)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"ruta_mapa_{timestamp}.html"
    ruta_salida = os.path.join("MAPA FINAL", nombre_archivo)
    mapa.save(ruta_salida)
    print(f"🗺️ Mapa guardado como '{ruta_salida}'. Ábrelo en un navegador para verlo.")

    # Devuelve el path
    return ruta_salida

def estimar_carga_para_ruta(origen, destino, hora="09:00", ES_ENTRE_SEMANA=1,
                             TEMPERATURA=12.0, HUMEDAD=70.0, PRESION=946.0,
                             RADIACION=150.0, PRECIPITA_BINARIA=0,
                             VELOCIDAD_VIENTO=1.5, DIR_VIENTO=180,
                             df_coordenadas_trafico=None):
    print("🚦 Iniciando estimación de carga para la ruta")
    if df_coordenadas_trafico is None:
        df_coordenadas_trafico = df_coordenadas_trafico_default
        print("📂 Usando df_coordenadas_trafico por defecto")

    sin_hora, cos_hora = hora_a_sin_cos(hora)
    TEMPERATURA = usar_valor_o_defecto(TEMPERATURA, "TEMPERATURA")
    HUMEDAD = usar_valor_o_defecto(HUMEDAD, "HUMEDAD")
    PRESION = usar_valor_o_defecto(PRESION, "PRESION")
    RADIACION = usar_valor_o_defecto(RADIACION, "RADIACION")
    VELOCIDAD_VIENTO = usar_valor_o_defecto(VELOCIDAD_VIENTO, "VELOCIDAD_VIENTO")
    DIR_VIENTO = usar_valor_o_defecto(DIR_VIENTO, "DIR_VIENTO")
    sin_dir_viento, cos_dir_viento = direccion_viento_a_sin_cos(DIR_VIENTO)

    contexto = {
        "HORA_sin": sin_hora,
        "HORA_cos": cos_hora,
        "ES_ENTRE_SEMANA": ES_ENTRE_SEMANA,
        "TEMPERATURA (ºC)": normalizar_variable("TEMPERATURA (ºC)", TEMPERATURA),
        "HUMEDAD RELATIVA (%)": normalizar_variable("HUMEDAD RELATIVA (%)", HUMEDAD),
        "PRESION BARIOMETRICA (mb)": normalizar_variable("PRESION BARIOMETRICA (mb)", PRESION),
        "RADIACION SOLAR (W/m2)": normalizar_variable("RADIACION SOLAR (W/m2)", RADIACION),
        "PRECIPITA_BINARIA": PRECIPITA_BINARIA,
        "VELOCIDAD VIENTO (m/s)": normalizar_variable("VELOCIDAD VIENTO (m/s)", VELOCIDAD_VIENTO),
        "DIR_VIENTO_sin": sin_dir_viento,
        "DIR_VIENTO_cos": cos_dir_viento
    }

    coordenadas_ruta = obtener_coordenadas_ruta(origen, destino)
    if not isinstance(coordenadas_ruta, list):
        raise ValueError(f"❌ Error al obtener coordenadas: {coordenadas_ruta}")

    bounding_box = obtener_bounding_box(coordenadas_ruta)
    print("📦 Bounding box construida:", bounding_box)

    df_filtrado = filtrar_puntos_medicion(df_coordenadas_trafico, bounding_box)
    print("🧪 Puntos filtrados:", len(df_filtrado))

    puntos_cercanos = encontrar_puntos_de_medicion(coordenadas_ruta, df_filtrado)
    puntos_con_predicciones = predecir_para_puntos(puntos_cercanos, contexto)
    ruta_mapa = visualizar_ruta(coordenadas_ruta, puntos_con_predicciones)
    print("🎯 Estimación finalizada")
    return coordenadas_ruta, puntos_con_predicciones, ruta_mapa
