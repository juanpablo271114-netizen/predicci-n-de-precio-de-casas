import os
import sys
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st


# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

st.set_page_config(
    page_title="Predicción Precio de Vivienda",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# ESTILOS VISUALES
# =====================================================

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
        padding: 36px;
        border-radius: 28px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.20);
        color: white;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #d1d5db;
    }

    .section-card {
        background: white;
        padding: 26px;
        border-radius: 24px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 850;
        color: #111827;
        margin-bottom: 5px;
    }

    .section-caption {
        color: #6b7280;
        font-size: 15px;
    }

    .prediction-box {
        background: linear-gradient(135deg, #064e3b 0%, #047857 65%, #10b981 100%);
        padding: 34px;
        border-radius: 28px;
        color: white;
        box-shadow: 0 18px 45px rgba(4, 120, 87, 0.30);
        text-align: center;
        margin-top: 20px;
    }

    .prediction-title {
        font-size: 16px;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #d1fae5;
    }

    .prediction-value {
        font-size: 48px;
        font-weight: 950;
        margin-top: 8px;
    }

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 22px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .metric-title {
        color: #6b7280;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: .7px;
    }

    .metric-value {
        color: #111827;
        font-size: 30px;
        font-weight: 900;
        margin-top: 6px;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #111827 0%, #374151 100%);
        color: white;
        border: none;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        font-size: 18px;
        font-weight: 800;
        box-shadow: 0 10px 25px rgba(17, 24, 39, 0.20);
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #000000 0%, #1f2937 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# FUNCIONES DE CONFIGURACIÓN
# =====================================================

def get_secret(key: str, default: str = "") -> str:
    """
    Busca primero en Streamlit Secrets y luego en variables de entorno.
    Esto permite ejecutar la app tanto localmente como en Streamlit Cloud.
    """
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


DATAROBOT_API_KEY = get_secret("DATAROBOT_API_KEY")
DATAROBOT_DEPLOYMENT_ID = get_secret("DATAROBOT_DEPLOYMENT_ID")
DATAROBOT_HOST = get_secret("DATAROBOT_HOST", "https://app.datarobot.com")


COLUMNAS_MODELO = [
    "total_habitaciones",
    "total_dormitorios",
    "poblacion",
    "hogares",
    "ingreso_mediano",
    "longitud",
    "latitud",
    "edad_mediana_vivienda",
    "proximidad_oceano"
]


def encontrar_columna_prediccion(df_resultado: pd.DataFrame, columnas_entrada: list) -> str:
    """
    Intenta identificar automáticamente la columna de predicción devuelta por DataRobot.
    """
    columnas_entrada = set(columnas_entrada)

    candidatas = [
        col for col in df_resultado.columns
        if col not in columnas_entrada
        and col.lower() not in ["prediction_status", "row_id"]
    ]

    candidatas_prioritarias = [
        col for col in candidatas
        if "prediction" in col.lower()
        or "pred" in col.lower()
        or "precio" in col.lower()
        or "valor" in col.lower()
    ]

    if candidatas_prioritarias:
        return candidatas_prioritarias[0]

    numericas = []
    for col in candidatas:
        serie = pd.to_numeric(df_resultado[col], errors="coerce")
        if serie.notna().any():
            numericas.append(col)

    if numericas:
        return numericas[0]

    if candidatas:
        return candidatas[0]

    return ""


def predecir_con_datarobot(df_entrada: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta predict.py usando un CSV temporal de entrada y genera un CSV temporal de salida.
    """

    ruta_predict = Path(__file__).parent / "predict.py"

    if not ruta_predict.exists():
        raise FileNotFoundError(
            "No se encontró predict.py en el repositorio. "
            "Debe estar en la misma carpeta que app.py."
        )

    if not DATAROBOT_API_KEY:
        raise ValueError(
            "Falta DATAROBOT_API_KEY. Configúralo en los Secrets de Streamlit Cloud."
        )

    if not DATAROBOT_DEPLOYMENT_ID:
        raise ValueError(
            "Falta DATAROBOT_DEPLOYMENT_ID. Configúralo en los Secrets de Streamlit Cloud."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "input.csv"
        output_csv = Path(tmpdir) / "output.csv"

        df_entrada.to_csv(input_csv, index=False)

        comando = [
            sys.executable,
            str(ruta_predict),
            str(input_csv),
            str(output_csv),
            DATAROBOT_DEPLOYMENT_ID,
            "--api_key",
            DATAROBOT_API_KEY,
            "--host",
            DATAROBOT_HOST,
            "--passthrough_columns_set",
            "--include_prediction_status",
            "--timeout",
            "600"
        ]

        resultado = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            timeout=900
        )

        if resultado.returncode != 0:
            raise RuntimeError(
                "DataRobot no pudo generar la predicción.\n\n"
                f"STDOUT:\n{resultado.stdout}\n\n"
                f"STDERR:\n{resultado.stderr}"
            )

        if not output_csv.exists():
            raise FileNotFoundError(
                "No se generó el archivo de salida de DataRobot."
            )

        return pd.read_csv(output_csv)


# =====================================================
# ENCABEZADO
# =====================================================

st.markdown("""
<div class="hero-card">
    <div class="hero-title">🏠 Predicción Inteligente del Precio de Vivienda</div>
    <div class="hero-subtitle">
        Interfaz conectada con DataRobot para estimar el precio de una casa a partir de variables habitacionales, geográficas y socioeconómicas.
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuración DataRobot")

    if DATAROBOT_API_KEY:
        st.success("API Key configurada")
    else:
        st.error("Falta API Key")

    if DATAROBOT_DEPLOYMENT_ID:
        st.success("Deployment ID configurado")
        st.caption(f"Deployment: `{DATAROBOT_DEPLOYMENT_ID[:8]}...`")
    else:
        st.error("Falta Deployment ID")

    st.markdown("---")
    st.markdown("### Variable objetivo")
    st.info("Precio de la vivienda")

    st.markdown("### Nota técnica")
    st.caption(
        "La variable `valor_mediano_vivienda` no se captura como entrada, "
        "porque corresponde a la variable objetivo que el modelo debe predecir."
    )


# =====================================================
# FORMULARIO
# =====================================================

col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Captura de variables</div>
        <div class="section-caption">
            Ingresa las características de la vivienda y su entorno. Estos datos serán enviados al modelo desplegado en DataRobot.
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
                max_value=30.0,
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
                    "<1H OCEAN",
                    "INLAND",
                    "ISLAND",
                    "NEAR BAY",
                    "NEAR OCEAN"
                ]
            )

        calcular = st.form_submit_button("Enviar a DataRobot y predecir")


# =====================================================
# DATAFRAME DE ENTRADA
# =====================================================

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

datos_usuario = datos_usuario[COLUMNAS_MODELO]


# =====================================================
# PANEL DERECHO
# =====================================================

with col2:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Registro a enviar</div>
        <div class="section-caption">
            Vista preliminar del registro que será enviado al deployment de DataRobot.
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


# =====================================================
# PREDICCIÓN
# =====================================================

st.markdown("---")

if calcular:
    try:
        with st.spinner("Enviando datos a DataRobot y generando predicción..."):
            resultado_datarobot = predecir_con_datarobot(datos_usuario)

        columna_prediccion = encontrar_columna_prediccion(
            resultado_datarobot,
            COLUMNAS_MODELO
        )

        st.success("Predicción generada correctamente")

        if columna_prediccion:
            valor_predicho = resultado_datarobot.loc[0, columna_prediccion]

            try:
                valor_predicho_num = float(valor_predicho)

                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-title">Precio estimado de la vivienda</div>
                    <div class="prediction-value">${valor_predicho_num:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            except Exception:
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-title">Predicción del modelo</div>
                    <div class="prediction-value">{valor_predicho}</div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.warning(
                "La predicción se generó, pero no pude identificar automáticamente "
                "la columna exacta del resultado."
            )

        st.markdown("### Resultado completo devuelto por DataRobot")
        st.dataframe(resultado_datarobot, use_container_width=True)

        csv_resultado = resultado_datarobot.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Descargar resultado CSV",
            data=csv_resultado,
            file_name="prediccion_datarobot.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("Ocurrió un error al ejecutar la predicción con DataRobot.")
        st.code(str(e))
