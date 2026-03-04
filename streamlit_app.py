
import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown

st.set_page_config(page_title="FruitScan", page_icon="🍊", layout="centered", initial_sidebar_state="expanded")

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

def descargar_modelo(nombre, file_id):
    ruta = os.path.join(MODELOS_DIR, f"{nombre}.tflite")
    if not os.path.exists(ruta):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, ruta, quiet=False)
    return ruta

@st.cache_resource(show_spinner="Descargando y cargando modelo...")
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

    # Obtener las dos salidas
    salidas = [interpreter.get_tensor(o['index'])[0] for o in output_details]
    return salidas[0], salidas[1]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    nombre_modelo = st.selectbox("Modelo", list(MODELOS_DRIVE.keys()), index=0)
    st.markdown("---")
    st.markdown("**Frutas soportadas**")
    for f in CLASES_FRUTA:
        st.markdown(f"• {EMOJIS_FRUTA.get(f,'🍑')} {f}")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🍊 FruitScan")
st.caption("Clasificador multitarea — Fruta & Estado")
st.info("ℹ️ La primera predicción descarga el modelo (~10-30MB). Las siguientes son instantáneas.")

# ── Upload ─────────────────────────────────────────────────────────────────────
archivo = st.file_uploader("Sube una imagen de fruta", type=["jpg","jpeg","png","webp"])

if archivo:
    imagen = Image.open(archivo)
    col1, col2 = st.columns(2)
    with col1:
        st.image(imagen, use_container_width=True, caption=archivo.name)
    with col2:
        with st.spinner("Analizando..."):
            try:
                interpreter = cargar_modelo(nombre_modelo)
                pred_fruta, pred_estado = predecir(interpreter, imagen)

                fruta  = CLASES_FRUTA[int(np.argmax(pred_fruta))]
                estado = CLASES_ESTADO[int(np.argmax(pred_estado))]
                emoji  = EMOJIS_FRUTA.get(fruta, "🍑")
                icono  = "✅" if estado == "Fresca" else "🔴"
                conf_f = float(np.max(pred_fruta)) * 100
                conf_e = float(np.max(pred_estado)) * 100

                st.markdown(f"## {emoji} {fruta}")
                st.markdown(f"### {icono} {estado}")
                st.metric("Confianza fruta",  f"{conf_f:.1f}%")
                st.metric("Confianza estado", f"{conf_e:.1f}%")
                st.caption(f"Modelo: {nombre_modelo} finetuned")
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Probabilidad por fruta**")
        for n, p in sorted(zip(CLASES_FRUTA, pred_fruta), key=lambda x: x[1], reverse=True):
            st.markdown(f"<span style='font-size:.75rem;color:#888'>{n}</span>", unsafe_allow_html=True)
            st.progress(float(p), text=f"{float(p)*100:.1f}%")
    with col4:
        st.markdown("**Estado**")
        for n, p in zip(CLASES_ESTADO, pred_estado):
            st.markdown(f"<span style='font-size:.75rem;color:#888'>{n}</span>", unsafe_allow_html=True)
            st.progress(float(p), text=f"{float(p)*100:.1f}%")
else:
    st.info("👆 Sube una imagen para comenzar el análisis.")
