import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================
# CONFIGURACIÓN GENERAL
# =========================

st.set_page_config(
    page_title="Predicción Precio de Vivienda",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# ESTILOS CSS
# =========================

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 50%, #374151 100%);
        padding: 35px;
        border-radius: 28px;
        box-shadow: 0 20px 45px rgba(0,0,0,0.18);
        color: white;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #d1d5db;
        margin-top: 0px;
    }

    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 24px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .metric-title {
        font-size: 14px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        font-size: 34px;
        font-weight: 800;
        color: #111827;
        margin-top: 8px;
    }

    .section-card {
        background: white;
        padding: 28px;
        border-radius: 24px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }

    .section-caption {
        color: #6b7280;
        font-size: 15px;
        margin-bottom: 18px;
    }

    .prediction-box {
        background: linear-gradient(135deg, #064e3b 0%, #047857 60%, #10b981 100%);
        padding: 32px;
        border-radius: 28px;
        color: white;
        box-shadow: 0 18px 45px rgba(4, 120, 87, 0.28);
        text-align: center;
    }

    .prediction-title {
        font-size: 16px;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #d1fae5;
    }

    .prediction-value {
        font-size: 46px;
        font-weight: 900;
        margin-top: 8px;
    }

    .warning-box {
        background: #fff7ed;
        color: #9a3412;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #fed7aa;
        font-size: 15px;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #111827 0%, #374151 100%);
        color: white;
        border: none;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        font-size: 18px;
        font-weight: 700;
        box-shadow: 0 10px 25px rgba(17, 24, 39, 0.18);
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #000000 0%, #1f2937 100%);
        color: white;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CARGA DEL MODELO
# =========================

@st.cache_resource
def cargar_modelo():
    """
    Ajusta el nombre del archivo según tu modelo entrenado.
    El modelo puede ser RandomForest, XGBoost, LinearRegression, Pipeline, etc.
    """
    ruta_modelo = Path("modelo_precio_casas.pkl")

    if ruta_modelo.exists():
        return joblib.load(ruta_modelo)

    return None


modelo = cargar_modelo()

# =========================
# ENCABEZADO
# =========================

st.markdown("""
<div class="hero-card">
    <div class="hero-title">🏠 Predicción Inteligente del Precio de Vivienda</div>
    <div class="hero-subtitle">
        Interfaz de captura de variables para estimar el valor de una casa usando un modelo predictivo de IA.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown("## ⚙️ Panel del modelo")

    if modelo is not None:
        st.success("Modelo cargado correctamente")
    else:
        st.warning("No se encontró el archivo modelo_precio_casas.pkl")

    st.markdown("---")
    st.markdown("""
    **Variable objetivo:**  
    Precio estimado de la vivienda.

    **Nota técnica:**  
    `valor_mediano_vivienda` no debe usarse como entrada si representa el precio real de la casa.
    """)

# =========================
# FORMULARIO PRINCIPAL
# =========================

col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Captura de características de la vivienda</div>
        <div class="section-caption">
            Ingresa los valores que describen la zona, composición habitacional y características socioeconómicas.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("formulario_prediccion"):

        st.markdown("### 🏘️ Características habitacionales")

        c1, c2, c3 = st.columns(3)

        with c1:
            total_habitaciones = st.number_input(
                "Total habitaciones",
                min_value=1,
                max_value=50000,
                value=2500,
                step=50
            )

        with c2:
            total_dormitorios = st.number_input(
                "Total dormitorios",
                min_value=1,
                max_value=20000,
                value=500,
                step=25
            )

        with c3:
            edad_mediana_vivienda = st.number_input(
                "Edad mediana vivienda",
                min_value=1,
                max_value=100,
                value=30,
                step=1
            )

        st.markdown("### 👥 Variables poblacionales")

        c4, c5, c6 = st.columns(3)

        with c4:
            poblacion = st.number_input(
                "Población",
                min_value=1,
                max_value=100000,
                value=1500,
                step=50
            )

        with c5:
            hogares = st.number_input(
                "Hogares",
                min_value=1,
                max_value=50000,
                value=600,
                step=25
            )

        with c6:
            ingreso_mediano = st.number_input(
                "Ingreso mediano",
                min_value=0.0,
                max_value=20.0,
                value=3.5,
                step=0.1,
                format="%.2f"
            )

        st.markdown("### 📍 Ubicación geográfica")

        c7, c8, c9 = st.columns(3)

        with c7:
            longitud = st.number_input(
                "Longitud",
                min_value=-180.0,
                max_value=180.0,
                value=-122.23,
                step=0.01,
                format="%.4f"
            )

        with c8:
            latitud = st.number_input(
                "Latitud",
                min_value=-90.0,
                max_value=90.0,
                value=37.88,
                step=0.01,
                format="%.4f"
            )

        with c9:
            proximidad_oceano = st.selectbox(
                "Proximidad al océano",
                [
                    "NEAR BAY",
                    "<1H OCEAN",
                    "INLAND",
                    "NEAR OCEAN",
                    "ISLAND"
                ]
            )

        calcular = st.form_submit_button("Calcular precio estimado")

# =========================
# DATOS CAPTURADOS
# =========================

datos_usuario = pd.DataFrame([{
    "total_habitaciones": total_habitaciones,
    "total_dormitorios": total_dormitorios,
    "poblacion": poblacion,
    "hogares": hogares,
    "ingreso_mediano": ingreso_mediano,
    "longitud": longitud,
    "latitud": latitud,
    "edad_mediana_vivienda": edad_mediana_vivienda,
    "proximidad_oceano": proximidad_oceano
}])

# =========================
# PANEL DERECHO
# =========================

with col2:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Resumen de variables</div>
        <div class="section-caption">
            Vista preliminar del registro que será enviado al modelo predictivo.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(datos_usuario, use_container_width=True)

    habitaciones_por_hogar = total_habitaciones / hogares
    dormitorios_por_hogar = total_dormitorios / hogares
    poblacion_por_hogar = poblacion / hogares

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Hab. por hogar</div>
            <div class="metric-value">{habitaciones_por_hogar:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Dorm. por hogar</div>
            <div class="metric-value">{dormitorios_por_hogar:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Pob. por hogar</div>
            <div class="metric-value">{poblacion_por_hogar:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# PREDICCIÓN
# =========================

st.markdown("---")

if calcular:

    if modelo is None:
        st.markdown("""
        <div class="warning-box">
            No se encontró un modelo entrenado en la carpeta del proyecto. 
            Guarda tu modelo con el nombre <b>modelo_precio_casas.pkl</b> para activar la predicción.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Registro capturado")
        st.dataframe(datos_usuario, use_container_width=True)

    else:
        try:
            prediccion = modelo.predict(datos_usuario)[0]

            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-title">Precio estimado de la vivienda</div>
                <div class="prediction-value">${prediccion:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error("Ocurrió un error al realizar la predicción.")
            st.code(str(e))

            st.markdown("""
            Posibles causas:
            - El modelo fue entrenado con nombres de columnas diferentes.
            - La variable categórica `proximidad_oceano` necesita codificación.
            - El modelo no fue guardado como Pipeline.
            """)

# =========================
# RECOMENDACIÓN FINAL
# =========================

st.markdown("""
<br>
<div class="section-card">
    <div class="section-title">Recomendación técnica</div>
    <div class="section-caption">
        Para evitar errores al predecir, lo ideal es guardar el modelo como un Pipeline que incluya:
        imputación de datos, escalamiento numérico, codificación de variables categóricas y el algoritmo predictivo.
    </div>
</div>
""", unsafe_allow_html=True)
