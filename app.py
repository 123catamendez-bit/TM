import streamlit as st
import cv2
import numpy as np
# from PIL import Image   # ← descomenta cuando tengas la imagen
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(page_title="🛰️ Explorador Visual | IA Galáctica", layout="centered")

# --- FONDO GALÁCTICO ---
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0b0f1a;
    background-image: radial-gradient(circle at 20% 20%, #16213e, #0b0f1a);
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background-color: #1a1f2e;
    color: #ffffff;
}
h1, h2, h3, p, label {
    color: #e0e0e0;
    font-family: 'Trebuchet MS', sans-serif;
}
.stButton>button {
    background-color: #25314d;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5em 1em;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- INFO DEL SISTEMA ---
st.markdown(f"<p style='text-align:right; color:#8f9bb3;'>🧩 Python version: {platform.python_version()}</p>", unsafe_allow_html=True)

# --- CARGA DEL MODELO ---
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# --- TÍTULO Y PRESENTACIÓN ---
st.title("🪐 Explorador Visual — IA Galáctica")
st.markdown("Analiza y clasifica objetos capturados por tu **cámara interestelar** utilizando un modelo entrenado en *Teachable Machine*. 🌌")

# --- IMAGEN DE PORTADA ---
# image = Image.open("explorador_visual.jpg")
# st.image(image, caption="🔭 Explorando las formas del universo", use_column_width=True)

# --- SIDEBAR ---
with st.sidebar:
    st.subheader("⚙️ Panel de Exploración")
    st.write("Toma una foto y deja que la **IA Galáctica** identifique lo que ve.")
    st.info("Modelo: `keras_model.h5`")

# --- CAPTURA DE IMAGEN ---
img_file_buffer = st.camera_input("📸 Captura tu imagen galáctica")

# --- PROCESAMIENTO ---
if img_file_buffer is not None:
    st.markdown("🧠 Analizando imagen... espere un momento")

    # Leer la imagen
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)

    # Convertir a arreglo numpy
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Inferencia
    prediction = model.predict(data)
    print(prediction)

    # --- RESULTADOS ---
    st.markdown("---")
    st.subheader("🔍 Resultado del análisis:")

    if prediction[0][0] > 0.5:
        st.success(f"🧑‍🚀 Objeto detectado: **Cata** 🌠 \nProbabilidad: {round(prediction[0][0]*100, 2)}%")
    elif prediction[0][1] > 0.5:
        st.success(f"📱 Objeto detectado: **Celular** ☄️ \nProbabilidad: {round(prediction[0][1]*100, 2)}%")
    elif prediction[0][2] > 0.5:
        st.success(f"🤚 Objeto detectado: **Mano** 🚀 \nProbabilidad: {round(prediction[0][2]*100, 2)}%")
    else:
        st.warning("⚠️ No se logró identificar el objeto con suficiente certeza. Intenta otra imagen o ajuste de iluminación.")

# --- PIE DE PÁGINA ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:#8f9bb3;'>🌌 Proyecto IA Galáctica · Explorador Visual con Teachable Machine · 2025</p>", unsafe_allow_html=True)
