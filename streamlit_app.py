import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown

st.set_page_config(
    page_title="FruitScan",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

/* Fondo general */
[data-testid="stAppViewContainer"] {
    background: #080b0f;
}
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2530;
}
[data-testid="stHeader"] { background: transparent; }

/* Tipografía global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e8edf2;
}

/* Título principal */
.main-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    letter-spacing: 0.08em;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 0;
}
.main-title span { color: #4ade80; }
.main-subtitle {
    font-size: 0.78rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 2rem;
}

/* Card de resultado */
.result-card {
    background: linear-gradient(135deg, #0d1117 0%, #111827 100%);
    border: 1px solid #1e2d3d;
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 8px;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
}
.result-fruta {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 0.05em;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 4px;
}
.result-fruta .emoji { font-size: 2.8rem; margin-right: 8px; }
.result-estado-fresh  {
    font-size: 1rem; font-weight: 500;
    color: #4ade80;
    background: rgba(74,222,128,.1);
    border: 1px solid rgba(74,222,128,.3);
    display: inline-block;
    padding: 4px 16px;
    border-radius: 100px;
    margin-bottom: 20px;
}
.result-estado-rotten {
    font-size: 1rem; font-weight: 500;
    color: #f87171;
    background: rgba(248,113,113,.1);
    border: 1px solid rgba(248,113,113,.3);
    display: inline-block;
    padding: 4px 16px;
    border-radius: 100px;
    margin-bottom: 20px;
}
.conf-row {
    display: flex; gap: 24px; margin-top: 8px;
}
.conf-box {
    flex: 1;
    background: #0a0e14;
    border: 1px solid #1e2530;
    border-radius: 10px;
    padding: 12px 16px;
}
.conf-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 4px;
}
.conf-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 0.05em;
    color: #4ade80;
}

/* Barras de probabilidad */
.bar-section-title {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 12px;
    margin-top: 20px;
}
.bar-row-custom {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.bar-name {
    width: 120px;
    font-size: 0.78rem;
    color: #8899a6;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.bar-track {
    flex: 1;
    height: 5px;
    background: #1e2530;
    border-radius: 3px;
    overflow: hidden;
}
.bar-fill-green  { height: 100%; border-radius: 3px; background: #4ade80; }
.bar-fill-cyan   { height: 100%; border-radius: 3px; background: #22d3ee; }
.bar-fill-red    { height: 100%; border-radius: 3px; background: #f87171; }
.bar-pct {
    width: 44px;
    font-size: 0.72rem;
    color: #4a5568;
    text-align: right;
    flex-shrink: 0;
    font-variant-numeric: tabular-nums;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #0d1117;
    border: 2px dashed #1e2d3d;
    border-radius: 12px;
    padding: 8px;
}
[data-testid="stFileUploader"]:hover {
    border-color: #4ade80;
}

/* Modelo tag */
.model-tag {
    font-size: 0.68rem;
    color: #2d3748;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 16px;
}

/* Sidebar */
.sidebar-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 8px;
}
.fruit-chip {
    display: inline-block;
    font-size: 0.72rem;
    color: #8899a6;
    background: #0d1117;
    border: 1px solid #1e2530;
    border-radius: 100px;
    padding: 3px 10px;
    margin: 3px 2px;
}

/* Ocultar elementos de Streamlit */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constantes ─────────────────────────────────────────────────────────────────
MODELOS_DRIVE = {
    "InceptionV3": "1BYrUWEhK_4NtXY_SKH2mpKTJ_uusHYOT",
    "MobileNetV2": "1MGM1f8c46f07j1K_UUllfoplhjCBhQCp",
    "VGG16":       "1g7EhijZb7zL_VxwEPYeJ54Sny-bhII1m",
}

CLASES_FRUTA  = ["Banano", "Fresa", "Limón", "Lulo", "Mango", "Naranja", "Tomate", "Tomate de Árbol"]
CLASES_ESTADO = ["Fresca", "Podrida"]
EMOJIS_FRUTA  = {"Banano":"🍌","Fresa":"🍓","Limón":"🍋","Lulo":"🟠","Mango":"🥭","Naranja":"🍊","Tomate":"🍅","Tomate de Árbol":"🍅"}

MODELOS_DIR = "/tmp/modelos"
os.makedirs(MODELOS_DIR, exist_ok=True)

# ── Funciones ──────────────────────────────────────────────────────────────────
def descargar_modelo(nombre, file_id):
    ruta = os.path.join(MODELOS_DIR, f"{nombre}.tflite")
    if not os.path.exists(ruta):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, ruta, quiet=False)
    return ruta

@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo(nombre):
    ruta = descargar_modelo(nombre, MODELOS_DRIVE[nombre])
    interpreter = tf.lite.Interpreter(model_path=ruta)
    interpreter.allocate_tensors()
    return interpreter

def predecir(interpreter, imagen):
    img = imagen.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    salidas = [interpreter.get_tensor(o['index'])[0] for o in output_details]
    return salidas[0], salidas[1]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:2.2rem">FRUIT<span>SCAN</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle" style="margin-bottom:1.5rem">Clasificador de frutas</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Modelo</div>', unsafe_allow_html=True)
    nombre_modelo = st.selectbox("", list(MODELOS_DRIVE.keys()), index=0, label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="sidebar-label">Frutas soportadas</div>', unsafe_allow_html=True)
    chips = "".join([f'<span class="fruit-chip">{EMOJIS_FRUTA.get(f,"🍑")} {f}</span>' for f in CLASES_FRUTA])
    st.markdown(chips, unsafe_allow_html=True)

# ── Layout principal ───────────────────────────────────────────────────────────
st.markdown('<div class="main-title">FRUIT<span>SCAN</span></div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Clasificador multitarea · Fruta &amp; Estado</div>', unsafe_allow_html=True)

col_img, col_res = st.columns([1.1, 1], gap="large")

with col_img:
    if st.session_state.get("limpiar"):
        st.session_state["limpiar"] = False
        st.rerun()

    archivo = st.file_uploader("Sube una imagen de fruta", type=["jpg","jpeg","png","webp"], key="uploader")
    if archivo:
        imagen = Image.open(archivo)
        st.image(imagen, use_container_width=True)
        if st.button("🗑️ Quitar imagen", use_container_width=True):
            st.session_state["limpiar"] = True
            st.rerun()
    else:
        st.markdown("""
        <div style="aspect-ratio:1; background:#0d1117; border:2px dashed #1e2530;
             border-radius:16px; display:flex; align-items:center; justify-content:center;
             flex-direction:column; gap:12px; color:#2d3748; text-align:center; padding:40px;">
            <div style="font-size:3rem;">🍊</div>
            <div style="font-size:0.8rem; letter-spacing:.1em; text-transform:uppercase;">
                Arrastra o selecciona<br>una imagen
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_res:
    if archivo:
        with st.spinner("Analizando..."):
            try:
                interpreter = cargar_modelo(nombre_modelo)
                pred_fruta, pred_estado = predecir(interpreter, imagen)

                fruta  = CLASES_FRUTA[int(np.argmax(pred_fruta))]
                estado = CLASES_ESTADO[int(np.argmax(pred_estado))]
                emoji  = EMOJIS_FRUTA.get(fruta, "🍑")
                conf_f = float(np.max(pred_fruta)) * 100
                conf_e = float(np.max(pred_estado)) * 100
                estado_class = "result-estado-fresh" if estado == "Fresca" else "result-estado-rotten"
                icono = "✅" if estado == "Fresca" else "🔴"

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-fruta"><span class="emoji">{emoji}</span>{fruta}</div>
                    <div class="{estado_class}">{icono} {estado}</div>
                    <div class="conf-row">
                        <div class="conf-box">
                            <div class="conf-label">Confianza fruta</div>
                            <div class="conf-value">{conf_f:.1f}%</div>
                        </div>
                        <div class="conf-box">
                            <div class="conf-label">Confianza estado</div>
                            <div class="conf-value">{conf_e:.1f}%</div>
                        </div>
                    </div>
                    <div class="model-tag">Modelo: {nombre_modelo} finetuned</div>
                </div>
                """, unsafe_allow_html=True)

                # Barras fruta
                st.markdown('<div class="bar-section-title">Probabilidad por fruta</div>', unsafe_allow_html=True)
                for n, p in sorted(zip(CLASES_FRUTA, pred_fruta), key=lambda x: x[1], reverse=True):
                    pct = float(p) * 100
                    fill = "bar-fill-green" if n == fruta else "bar-fill-cyan"
                    st.markdown(f"""
                    <div class="bar-row-custom">
                        <span class="bar-name">{n}</span>
                        <div class="bar-track"><div class="{fill}" style="width:{pct:.1f}%"></div></div>
                        <span class="bar-pct">{pct:.1f}%</span>
                    </div>""", unsafe_allow_html=True)

                # Barras estado
                st.markdown('<div class="bar-section-title">Estado</div>', unsafe_allow_html=True)
                for n, p in zip(CLASES_ESTADO, pred_estado):
                    pct = float(p) * 100
                    fill = "bar-fill-green" if n == "Fresca" else "bar-fill-red"
                    st.markdown(f"""
                    <div class="bar-row-custom">
                        <span class="bar-name">{n}</span>
                        <div class="bar-track"><div class="{fill}" style="width:{pct:.1f}%"></div></div>
                        <span class="bar-pct">{pct:.1f}%</span>
                    </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.markdown("""
        <div style="padding: 40px 0; color:#2d3748; font-size:0.85rem; line-height:2;">
            ← Sube una imagen para<br>comenzar el análisis.
        </div>
        """, unsafe_allow_html=True)

