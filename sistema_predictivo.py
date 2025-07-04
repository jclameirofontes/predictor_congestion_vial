import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from lightgbm import LGBMRegressor
import googlemaps
import requests
from geopy.distance import geodesic
import folium
from folium import DivIcon
from shapely.geometry import Point
from scipy.spatial import cKDTree
from datetime import datetime, timedelta
from PIL import Image


API_KEY = "AIzaSyCvHe4yktaXZzR460Ki8nGUET-9Arm3Lps"
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

def usar_valor_o_defecto(valor, nombre_variable, hora_entera=None):
    if valor is None or str(valor).strip().upper() in ["NA", "NAN", ""]:
        if nombre_variable == "RADIACION" and hora_entera is not None:
            if hora_entera == 0 or hora_entera == 1 or hora_entera == 2 or hora_entera == 3 or hora_entera == 4 or hora_entera == 5 or hora_entera == 6 or hora_entera == 7 or hora_entera == 19 or hora_entera == 20 or hora_entera == 21 or hora_entera == 22 or hora_entera == 23:
                return 1.0
            elif hora_entera == 8:
                return 23.0
            elif hora_entera == 9:
                return 135.0
            elif hora_entera == 10:
                return 289.0
            elif hora_entera == 11:
                return 406.0
            elif hora_entera == 12:
                return 486.0
            elif hora_entera == 13:
                return 489.0
            elif hora_entera == 14:
                return 430.0
            elif hora_entera == 15:
                return 326.0
            elif hora_entera == 16:
                return 185.0
            elif hora_entera == 17:
                return 60.0
            elif hora_entera == 18:
                return 5.0
        valores_por_defecto = {
            "TEMPERATURA": 12.0,
            "HUMEDAD": 70.0,
            "PRESION": 946.0,
            "RADIACION": 150.0,
            "VELOCIDAD_VIENTO": 1.5,
            "DIR_VIENTO": 180
        }
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
        puntos_cercanos[id_punto]["id_target"] = id_target
    print("✅ Predicciones realizadas para:", list(predicciones.keys()))
    return puntos_cercanos


def obtener_tiempo_sin_trafico(origen, destino):
    from datetime import datetime
    try:
        hoy = datetime.now()

        dias_hasta_domingo = (6 - hoy.weekday()) % 7
        if dias_hasta_domingo == 0:
            dias_hasta_domingo = 7
        
        proximo_domingo = hoy + timedelta(days=dias_hasta_domingo)
        hora_baja = proximo_domingo.replace(hour=4, minute=0, second=0, microsecond=0)

        directions = gmaps.directions(
            origen,
            destino,
            mode="driving",
            departure_time=hora_baja
        )

        if directions and directions[0].get("legs"):
            leg = directions[0]["legs"][0]
            tiempo_ideal = leg["duration_in_traffic"]["value"] / 60  # en minutos
            return tiempo_ideal
    except Exception as e:
        print(f"❌ Error al obtener tiempo sin tráfico (domingo 4AM): {e}")
    return None

def color_por_carga(carga):
    if carga <= 0.2:
        return "green"
    elif carga <= 0.3:
        return "gold"
    elif carga <= 0.5:
        return "orange"
    else:
        return "red"
            
# Función para boxplot sobrepuesto con predicciones
def generar_boxplot_superpuesto(cargas, ruta_base="carga_coloreada.png", salida="boxplot_superpuesto.png"):
    colores = [color_por_carga(c) for c in cargas]

    img = Image.open(ruta_base)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(img, extent=[0, 1, 0, 1], aspect='auto')
    ax.axis("off")

    # Segundo eje para boxplot
    ax2 = ax.inset_axes([0.103, 0.54, 0.862, 0.123])
    ax2.set_ylim(0, 2)
    ax2.set_xlim(0, 1)
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_ylim(-1, 2)
    ax2.axis("off")

    # Coordenada vertical centrada (0.5 es medio)
    y_pos = 0.1

    box = ax2.boxplot(
        [cargas],
        positions=[y_pos],
        vert=False,
        widths=2,
        whis=[0, 100],  # fuerza que los whiskers vayan del mínimo al máximo
        patch_artist=True,
        boxprops=dict(facecolor='lightgray', color='black', linewidth=1.5),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black')
    )

    for c, col in zip(cargas, colores):
        ax2.plot(c, y_pos, 'o', color=col, markersize=8, markeredgecolor='black')

        # Texto descriptivo sobre la mediana
    # Texto descriptivo sobre la mediana
    mediana = np.median(cargas)
    if mediana <= 0.2:
        texto = "CARGA BAJA"
    elif mediana <= 0.3:
        texto = "CARGA MODERADA"
    elif mediana <= 0.5:
        texto = "CARGA ALTA"
    else:
        texto = "CARGA MUY ALTA"

    color_margen = color_por_carga(mediana)

    ax2.text(mediana, y_pos + 2, texto,
            ha='center', va='bottom', fontsize=18,
            bbox=dict(facecolor=color_margen, edgecolor='black', boxstyle='round,pad=0.3'),
            color='black', weight='bold')

    plt.tight_layout()
    fig.savefig(salida, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Boxplot generado correctamente como: {salida}")


def visualizar_ruta(coordenadas_ruta, puntos_medicion, tiempo_sin_trafico):
    import numpy as np
    import folium
    from shapely.geometry import Point
    import matplotlib.pyplot as plt
    from PIL import Image
    import base64

    if not coordenadas_ruta or isinstance(coordenadas_ruta, str):
        print("No se puede visualizar la ruta: No hay datos válidos.")
        return

    puntos_medicion = {
        id_punto: datos for id_punto, datos in puntos_medicion.items()
        if datos.get("prediccion") is not None
    }

    mapa = folium.Map(location=coordenadas_ruta[0], zoom_start=14)

    puntos_con_carga = [
        (Point(info["longitud"], info["latitud"]), info.get("prediccion", 0.0))
        for info in puntos_medicion.values()
    ]

    def color_por_carga(carga):
        if carga <= 0.2:
            return "green"
        elif carga <= 0.3:
            return "gold"
        elif carga <= 0.5:
            return "orange"
        else:
            return "red"

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
        id_target = datos.get("id_target")
        if pred is not None:
            color = color_por_carga(pred)
            carga_media_text = f"{id_target:.3f}" if id_target is not None else "N/A"
            popup_texto = f"""
                <div style='font-size: 16px; line-height: 1.5;'>
                    <b style='font-size:18px'>{datos['nombre']}</b><br>
                    <b>ID:</b> {id_punto}<br>
                    <b>Grupo:</b> {grupo}<br>
                    <b>Carga media:</b> {carga_media_text}<br>
                    <div style='background-color:{color}; color:white; text-align:center;
                                padding:6px 0; font-weight:bold; margin-top:6px;
                                border:2px solid black; border-radius:6px; font-size:18px;'>
                        {pred:.3f}
                    </div>
                </div>
            """

            
            folium.Marker(
                location=(datos["latitud"], datos["longitud"]),
                popup=folium.Popup(popup_texto, max_width=400),  # más ancho
                icon=folium.DivIcon(html=f"""
                    <div style="
                        background-color:{color};
                        border-radius:50%;
                        color:white;
                        font-size:14pt;  /* más grande */
                        font-weight:bold;
                        text-align:center;
                        width:32px;
                        height:32px;
                        line-height:32px;
                        box-shadow: 0 0 3px #000;
                    ">
                        {grupo}
                    </div>""")
            ).add_to(mapa)


    folium.Marker(coordenadas_ruta[0], popup="Inicio", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(mapa)
    folium.Marker(coordenadas_ruta[-1], popup="Destino", icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")).add_to(mapa)

    carga_media = np.mean([datos["prediccion"] for datos in puntos_medicion.values()])

    predicciones = [datos["prediccion"] for datos in puntos_medicion.values()]
    predicciones_validas = [p for p in predicciones if p is not None]
    generar_boxplot_superpuesto(predicciones)

    with open("boxplot_superpuesto.png", "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    html_info = f"""
    <div style='position: fixed; top: 20px; left: 20px; z-index: 9999; background-color: white;
                border: 3px solid #222; padding: 16px; border-radius: 10px; font-size: 18px; width: 650px;'>
        <details open>
            <summary style="font-size:24px; font-weight:bold; cursor:pointer;">
                <div style="text-align:center;">📊 Información de Tráfico</div>
            </summary>
            <div style="margin-top: 12px; line-height: 1.6; text-align:center;">
                <div style="display:inline-block; padding:10px 18px; background-color:#f0f0f0;
                            border:1px solid #bbb; border-radius:8px; margin: 6px;">
                    ⏱️ <b>Tiempo sin tráfico:</b> <span style="font-size:20px;">{tiempo_sin_trafico:.1f} min</span>
                </div><br>
                <div style="display:inline-block; padding:10px 18px; background-color:{color_por_carga(carga_media)};
                            border:2px solid black; border-radius:8px; color:white; font-weight:bold; margin: 6px;">
                    🚗 <b>Carga media:</b> <span style="font-size:20px;">{carga_media:.2f}</span>
                </div>
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{img_base64}" style="width: 600px; margin-top: 16px; border: 1px solid #ccc;" />
                </div>
            </div>
        </details>
    </div>
    
    <div style='position: fixed; top: 20px; right: 20px; z-index: 9999; background-color: white;
                border: 3px solid #222; padding: 16px; border-radius: 10px; font-size: 18px; width: 240px;'>
        <details open>
            <summary style="font-size:20px; font-weight:bold; cursor:pointer;">
                <div style="text-align:center;">🎨 Leyenda</div>
            </summary>
            <div style="margin-top: 10px; line-height: 1.6; text-align:left;">
                <span style="color:green; font-weight:bold; font-size:18px;">●</span> <b>Baja</b> <0.2<br>
                <span style="color:gold; font-weight:bold; font-size:18px;">●</span> <b>Moderada</b> 0.2-0.3<br>
                <span style="color:orange; font-weight:bold; font-size:18px;">●</span> <b>Alta</b> 0.3-0.5<br>
                <span style="color:red; font-weight:bold; font-size:18px;">●</span> <b>Muy Alta</b> >0.5
            </div>
        </details>
    </div>
    """



    mapa.get_root().html.add_child(folium.Element(html_info))


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
   
    hora_entera = int(hora.split(":")[0])
    sin_hora, cos_hora = hora_a_sin_cos(hora)
    TEMPERATURA = usar_valor_o_defecto(TEMPERATURA, "TEMPERATURA")
    HUMEDAD = usar_valor_o_defecto(HUMEDAD, "HUMEDAD")
    PRESION = usar_valor_o_defecto(PRESION, "PRESION")
    RADIACION = usar_valor_o_defecto(RADIACION, "RADIACION", hora_entera=hora_entera)
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
    tiempo_sin_trafico = obtener_tiempo_sin_trafico(origen, destino)
    ruta_mapa = visualizar_ruta(coordenadas_ruta, puntos_con_predicciones, tiempo_sin_trafico)
    print("🎯 Estimación finalizada")
    return coordenadas_ruta, puntos_con_predicciones, ruta_mapa
