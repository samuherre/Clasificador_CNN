import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from utils import load_trained_model, predict_digit

st.title("🧠 MNIST Digit Classifier (CNN)")
st.write("Selecciona una imagen del dataset y predice el número.")

# Cargar datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Slider para elegir imagen
index = st.slider("Selecciona una imagen", 0, len(x_test)-1, 0)

image = x_test[index]
label = y_test[index]

# Mostrar imagen
st.subheader("Imagen seleccionada")
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.set_title(f"Etiqueta real: {label}")
ax.axis('off')
st.pyplot(fig)

# Botón de predicción
if st.button("Predecir"):
    model = load_trained_model()

    # Normalizar
    image_norm = image / 255.0

    pred, conf = predict_digit(model, image_norm)

    st.subheader("Resultado")
    st.write(f"Predicción: **{pred}**")
    st.write(f"Confianza: **{conf:.4f}**")

    if pred == label:
        st.success("✔ Predicción correcta")
    else:
        st.error("❌ Predicción incorrecta")
