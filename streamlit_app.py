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

# ── Inicializar tema ───────────────────────────────────────────────────────────
if "tema" not in st.session_state:
    st.session_state.tema = "dark"

tema = st.session_state.tema

# ── Variables de color por tema ────────────────────────────────────────────────
if tema == "dark":
    BG       = "#080c12"
    SURFACE  = "#111826"
    BORDER   = "#1e2d42"
    TEXT     = "#e8f0f8"
    TEXT2    = "#94a8bc"
    MUTED    = "#4a6070"
    ACCENT   = "#38bdf8"
    ACCENT2  = "#818cf8"
    FRESH    = "#34d399"
    ROTTEN   = "#fb7185"
    BTNTEXT  = "#080c12"
    GRAD1    = "rgba(56,189,248,.06)"
    GRAD2    = "rgba(129,140,248,.06)"
else:
    BG       = "#f0f5fb"
    SURFACE  = "#ffffff"
    BORDER   = "#d0dce8"
    TEXT     = "#0d1926"
    TEXT2    = "#3d5470"
    MUTED    = "#8099b0"
    ACCENT   = "#0284c7"
    ACCENT2  = "#6366f1"
    FRESH    = "#059669"
    ROTTEN   = "#e11d48"
    BTNTEXT  = "#ffffff"
    GRAD1    = "rgba(2,132,199,.05)"
    GRAD2    = "rgba(99,102,241,.05)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif !important; }}

[data-testid="stAppViewContainer"] {{
    background: {BG};
    background-image: radial-gradient(ellipse 60% 40% at 10% 10%, {GRAD1} 0%, transparent 60%),
                      radial-gradient(ellipse 40% 30% at 90% 80%, {GRAD2} 0%, transparent 60%);
}}
[data-testid="stSidebar"] {{
    background: {SURFACE} !important;
    border-right: 1px solid {BORDER};
}}
[data-testid="stHeader"] {{ background: transparent; }}

.main-title {{
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
    font-size: 3.2rem;
    letter-spacing: -0.03em;
    color: {ACCENT};
    line-height: 1;
    margin-bottom: 0;
}}
.main-title span {{ color: {TEXT}; }}
.main-subtitle {{
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 2rem;
}}

.result-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 8px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,.15);
}}
.result-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
}}
.result-fruta {{
    font-family: 'Outfit', sans-serif;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    color: {TEXT};
    line-height: 1;
    margin-bottom: 10px;
}}
.result-estado-fresh {{
    font-size: 1rem; font-weight: 600;
    color: {FRESH};
    background: rgba(52,211,153,.1);
    border: 1px solid {FRESH};
    display: inline-block;
    padding: 5px 18px;
    border-radius: 100px;
    margin-bottom: 20px;
}}
.result-estado-rotten {{
    font-size: 1rem; font-weight: 600;
    color: {ROTTEN};
    background: rgba(251,113,133,.1);
    border: 1px solid {ROTTEN};
    display: inline-block;
    padding: 5px 18px;
    border-radius: 100px;
    margin-bottom: 20px;
}}
.conf-row {{ display: flex; gap: 16px; margin-top: 10px; }}
.conf-box {{
    flex: 1;
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 14px 18px;
}}
.conf-label {{
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 4px;
    font-weight: 600;
}}
.conf-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: {ACCENT};
}}
.bar-section-title {{
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 12px;
    margin-top: 20px;
    font-weight: 600;
}}
.bar-row-c {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}}
.bar-name {{
    width: 120px;
    font-size: 0.76rem;
    color: {TEXT2};
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.bar-track {{
    flex: 1;
    height: 6px;
    background: {BORDER};
    border-radius: 4px;
    overflow: hidden;
}}
.bar-fill-a  {{ height:100%; border-radius:4px; background: linear-gradient(90deg, {ACCENT}, {ACCENT2}); }}
.bar-fill-b  {{ height:100%; border-radius:4px; background: {BORDER}; }}
.bar-fill-f  {{ height:100%; border-radius:4px; background: {FRESH}; }}
.bar-fill-r  {{ height:100%; border-radius:4px; background: {ROTTEN}; }}
.bar-pct {{
    width: 48px;
    font-size: 0.7rem;
    color: {MUTED};
    text-align: right;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}}
.info-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 28px;
    height: 100%;
}}
.info-big {{
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
    font-size: 1.8rem;
    line-height: 1.2;
    color: {ACCENT};
    margin-bottom: 12px;
}}
.info-desc {{
    font-size: 0.85rem;
    color: {TEXT2};
    line-height: 1.8;
    margin-bottom: 20px;
}}
.fruit-chip {{
    font-size: 0.7rem;
    padding: 4px 11px;
    border: 1px solid {BORDER};
    border-radius: 100px;
    color: {TEXT2};
    background: {BG};
    display: inline-block;
    margin: 3px 2px;
}}
.model-tag {{
    font-size: 0.65rem;
    color: {MUTED};
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-top: 14px;
    font-family: 'JetBrains Mono', monospace;
}}
[data-testid="stFileUploader"] {{
    background: {BG};
    border: 2px dashed {BORDER};
    border-radius: 12px;
}}
#MainMenu, footer {{ visibility: hidden; }}
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
    st.markdown(f'<div class="main-title" style="font-size:2rem">FRUIT<span>SCAN</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-subtitle">Clasificador de frutas</div>', unsafe_allow_html=True)

    # Toggle tema
    icono = "☀️" if tema == "dark" else "🌙"
    label = "Modo claro" if tema == "dark" else "Modo oscuro"
    if st.button(f"{icono} {label}", use_container_width=True):
        st.session_state.tema = "light" if tema == "dark" else "dark"
        st.rerun()

    st.markdown("---")
    st.markdown(f'<div style="font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:{MUTED};margin-bottom:10px;font-weight:600">Modelo</div>', unsafe_allow_html=True)
    nombre_modelo = st.selectbox("", list(MODELOS_DRIVE.keys()), index=0, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f'<div style="font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:{MUTED};margin-bottom:10px;font-weight:600">Frutas soportadas</div>', unsafe_allow_html=True)
    chips = "".join([f'<span class="fruit-chip">{EMOJIS_FRUTA.get(f,"🍑")} {f}</span>' for f in CLASES_FRUTA])
    st.markdown(chips, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="main-title">FRUIT<span>SCAN</span></div>', unsafe_allow_html=True)
st.markdown(f'<div class="main-subtitle">Clasificador multitarea &middot; Fruta &amp; Estado</div>', unsafe_allow_html=True)

col_img, col_res = st.columns([1.1, 1], gap="large")

# ── Columna imagen ─────────────────────────────────────────────────────────────
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
        st.markdown(f"""
        <div style="aspect-ratio:1;background:{SURFACE};border:2px dashed {BORDER};
             border-radius:16px;display:flex;align-items:center;justify-content:center;
             flex-direction:column;gap:12px;color:{MUTED};text-align:center;padding:40px;">
            <div style="font-size:3rem;filter:grayscale(.5)">🍊</div>
            <div style="font-size:0.78rem;letter-spacing:.1em;text-transform:uppercase;line-height:1.8">
                Arrastra o selecciona<br>una imagen
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Columna resultados ─────────────────────────────────────────────────────────
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
                es_fresca = estado == "Fresca"
                icono_e = "✅" if es_fresca else "🔴"
                estado_class = "result-estado-fresh" if es_fresca else "result-estado-rotten"

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-fruta">{emoji} {fruta}</div>
                    <div class="{estado_class}">{icono_e} {estado}</div>
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
                    fill = "bar-fill-a" if n == fruta else "bar-fill-b"
                    st.markdown(f"""
                    <div class="bar-row-c">
                        <span class="bar-name">{n}</span>
                        <div class="bar-track"><div class="{fill}" style="width:{pct:.1f}%"></div></div>
                        <span class="bar-pct">{pct:.1f}%</span>
                    </div>""", unsafe_allow_html=True)

                # Barras estado
                st.markdown('<div class="bar-section-title">Estado</div>', unsafe_allow_html=True)
                for n, p in zip(CLASES_ESTADO, pred_estado):
                    pct = float(p) * 100
                    fill = "bar-fill-f" if n == "Fresca" else "bar-fill-r"
                    st.markdown(f"""
                    <div class="bar-row-c">
                        <span class="bar-name">{n}</span>
                        <div class="bar-track"><div class="{fill}" style="width:{pct:.1f}%"></div></div>
                        <span class="bar-pct">{pct:.1f}%</span>
                    </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-big">Sube una foto<br>de una fruta.</div>
            <div class="info-desc">
                El modelo identificará la <strong style="color:{TEXT}">especie</strong> y determinará
                si está <strong style="color:{FRESH}">fresca</strong> o
                <strong style="color:{ROTTEN}">podrida</strong>.
            </div>
            <div style="font-size:.62rem;letter-spacing:.18em;text-transform:uppercase;color:{MUTED};margin-bottom:10px;font-weight:600">
                Frutas soportadas
            </div>
            {"".join([f'<span class="fruit-chip">{EMOJIS_FRUTA.get(f,"🍑")} {f}</span>' for f in CLASES_FRUTA])}
        </div>
        """, unsafe_allow_html=True)

