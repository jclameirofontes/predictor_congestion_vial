
import streamlit as st
from datetime import time
from sistema_predictivo import estimar_carga_para_ruta
import os
import folium
from streamlit.components.v1 import html

# Configuración de página
st.set_page_config(
    page_title="Predictor de Congestión Vial",
    layout="wide"
)

# Título y descripción
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("## 🚦 Predictor de Congestión Vial en Madrid")
    st.markdown("""
        Completa los campos para estimar la carga de tráfico.  
        Los campos marcados con *️⃣ son obligatorios.
    """)
    
    origen = st.text_input("📍 Dirección de Origen *", "")
    destino = st.text_input("🏁 Dirección de Destino *", "")
    hora = st.time_input("⏰ Hora de salida *", value=time(0, 0))
    semana = st.selectbox("¿Es entre semana? *", ["Sí", "No"])
    lluvia = st.selectbox("¿Está lloviendo? *", ["No", "Sí"])

    st.markdown("---")
    temp = st.text_input("🌡️ Temperatura (ºC) (entre 3.25 y 22.85)", "")
    humedad = st.text_input("💧 Humedad (%) (entre 33 y 100)", "")
    presion = st.text_input("📘 Presión (mb) (entre 933.43 y 968)", "")
    radiacion = st.text_input("☀️ Radiación solar (W/m2) (entre 0 y 705)", "")
    v_viento = st.text_input("🍃 Velocidad del viento (m/s) (entre 0 y 4.775)", "")
    dir_viento = st.text_input("🧭 Dirección del viento (º) (entre 0 y 359)", "")

st.markdown("---")

# Estilo personalizado para botón
st.markdown("""
<style>
div.stButton > button {
    background-color: #b5e7a0;
    color: black;
    font-weight: bold;
    font-size: 18px;
    border: 3px solid black;
    border-radius: 10px;
    padding: 0.5em 1.5em;
    width: 100%;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

estimar = st.button("▶️ Estimar congestión")

if estimar:
        if origen.strip() == "" or destino.strip() == "":
            st.error("Por favor, completa las direcciones de origen y destino.")
        else:
            with st.spinner("🔄 Procesando predicción..."):
                try:
                    semana_valor = 1 if semana == "Sí" else 0
                    lluvia_valor = 1 if lluvia == "Sí" else 0

                    _, puntos, ruta_html = estimar_carga_para_ruta(
                        origen=origen,
                        destino=destino,
                        hora=hora.strftime("%H:%M"),
                        ES_ENTRE_SEMANA=semana_valor,
                        TEMPERATURA=None if temp.upper() == "NA" or temp.strip() == "" else float(temp),
                        HUMEDAD=None if humedad.upper() == "NA" or humedad.strip() == "" else float(humedad),
                        PRESION=None if presion.upper() == "NA" or presion.strip() == "" else float(presion),
                        RADIACION=None if radiacion.upper() == "NA" or radiacion.strip() == "" else float(radiacion),
                        VELOCIDAD_VIENTO=None if v_viento.upper() == "NA" or v_viento.strip() == "" else float(v_viento),
                        DIR_VIENTO=None if dir_viento.upper() == "NA" or dir_viento.strip() == "" else float(dir_viento),
                        PRECIPITA_BINARIA=lluvia_valor,
                        df_coordenadas_trafico=None
                    )

                    if os.path.exists(ruta_html):
                        with open(ruta_html, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                            st.success("✅ Predicción completada. Mapa generado.")
                            with col2:
                                html(html_content, height=1600)
                    else:
                        st.error("No se encontró el archivo del mapa generado.")
                except Exception as e:
                    st.error(f"Error al estimar la congestión: {e}")

with col2:
    if not os.path.exists("mapa/MAPA FINAL/temp_map.html"):
        m = folium.Map(location=[40.4168, -3.7038], zoom_start=12)
        m.save("MAPA FINAL/temp_map.html")

    with open("MAPA FINAL/temp_map.html", 'r', encoding='utf-8') as f:
        default_map = f.read()
        html(default_map, height=1275)
