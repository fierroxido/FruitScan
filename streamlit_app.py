import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown

st.set_page_config(page_title="FruitScan", page_icon="🍊", layout="centered", initial_sidebar_state="expanded")

MODELOS_DRIVE = {
    "MobileNetV2_finetuned": "1HUduen1mR02tBU5Qqshd_uBr0gUzqXSy",
    "MobileNetV2_best":      "14ZUKtsMKC7C7o_drFpy_xii6vxGYc8Oc",
    "InceptionV3_finetuned": "1C9Ny3ZTKWZi24OqgLD9eal0rqz49mf57",
    "InceptionV3_best":      "1wpcCNg7ORR09_Awq_21x-NQlKkY2IVC9",
    "VGG16_finetuned":       "1-nBisjMKqhFPcccpI6bA9wy3uO--F6ln",
    "VGG16_best":            "1J4tSIWB78tV8oFY3UJ0rWR7NDa-P_mab",
}

CLASES_FRUTA  = ["Banano", "Fresa", "Limón", "Lulo", "Mango", "Naranja", "Tomate", "Tomate de Árbol"]
CLASES_ESTADO = ["Fresca", "Podrida"]
EMOJIS_FRUTA  = {"Banano":"🍌","Fresa":"🍓","Limón":"🍋","Lulo":"🟠","Mango":"🥭","Naranja":"🍊","Tomate":"🍅","Tomate de Árbol":"🍅"}

MODELOS_DIR = "modelos"
os.makedirs(MODELOS_DIR, exist_ok=True)

def descargar_modelo(nombre, file_id):
    ruta = os.path.join(MODELOS_DIR, f"{nombre}.h5")
    if not os.path.exists(ruta):
        with st.spinner(f"Descargando {nombre} desde Google Drive..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, ruta, quiet=False)
    return ruta

@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo(nombre):
    ruta = descargar_modelo(nombre, MODELOS_DRIVE[nombre])
    return tf.keras.models.load_model(ruta, compile=False)

def preprocesar(imagen):
    img = imagen.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    nombre_modelo = st.selectbox("Modelo", list(MODELOS_DRIVE.keys()), index=2)
    st.markdown("---")
    st.markdown("**Frutas soportadas**")
    for f in CLASES_FRUTA:
        st.markdown(f"• {EMOJIS_FRUTA.get(f,'🍑')} {f}")

st.title("🍊 FruitScan")
st.caption("Clasificador multitarea — Fruta & Estado")

archivo = st.file_uploader("Sube una imagen de fruta", type=["jpg","jpeg","png","webp"])

if archivo:
    imagen = Image.open(archivo)
    col1, col2 = st.columns(2)
    with col1:
        st.image(imagen, use_container_width=True, caption=archivo.name)
    with col2:
        with st.spinner("Analizando..."):
            try:
                modelo  = cargar_modelo(nombre_modelo)
                entrada = preprocesar(imagen)
                salida  = modelo.predict(entrada, verbose=0)
                if isinstance(salida, dict):
                    pred_fruta  = salida["salida_fruta"][0]
                    pred_estado = salida["salida_estado"][0]
                else:
                    pred_fruta  = salida[0][0]
                    pred_estado = salida[1][0]
                fruta  = CLASES_FRUTA[int(np.argmax(pred_fruta))]
                estado = CLASES_ESTADO[int(np.argmax(pred_estado))]
                emoji  = EMOJIS_FRUTA.get(fruta, "🍑")
                icono  = "✅" if estado == "Fresca" else "🔴"
                conf_f = float(pred_fruta[int(np.argmax(pred_fruta))]) * 100
                conf_e = float(pred_estado[int(np.argmax(pred_estado))]) * 100
                st.markdown(f"## {emoji} {fruta}")
                st.markdown(f"### {icono} {estado}")
                st.metric("Confianza fruta",  f"{conf_f:.1f}%")
                st.metric("Confianza estado", f"{conf_e:.1f}%")
                st.caption(f"Modelo: {nombre_modelo}")
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
