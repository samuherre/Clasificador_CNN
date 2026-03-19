import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ─────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Reconocimiento de Dígitos - CNN",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Reconocimiento de Dígitos con CNN")
st.markdown(
    "Selecciona una imagen del dataset **MNIST Digits (sklearn)** "
    "y la red neuronal convolucional predecirá el dígito."
)

# ─────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────
MODEL_PATH = "cnn_digits_model.keras"
IMG_SIZE   = 8   # sklearn digits son 8x8
EPOCHS     = 30
BATCH_SIZE = 32

# ─────────────────────────────────────────────
# Carga del dataset
# ─────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    """Carga y prepara el dataset Digits de sklearn."""
    digits = load_digits()
    X = digits.images                          # (1797, 8, 8)
    y = digits.target                          # (1797,)

    # Normalizar y añadir canal (para CNN)
    X = X / 16.0                               # valores originales 0-16
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)   # (N, 8, 8, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return digits, X, y, X_train, X_test, y_train, y_test

# ─────────────────────────────────────────────
# Definición del modelo CNN
# ─────────────────────────────────────────────
def construir_modelo():
    """Construye la arquitectura CNN."""
    modelo = models.Sequential([
        # Bloque convolucional 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloque convolucional 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Clasificador
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return modelo

# ─────────────────────────────────────────────
# Entrenamiento / carga del modelo
# ─────────────────────────────────────────────
@st.cache_resource
def obtener_modelo(X_train, y_train, X_test, y_test):
    """Entrena el modelo si no existe, o lo carga desde disco."""
    if os.path.exists(MODEL_PATH):
        modelo = tf.keras.models.load_model(MODEL_PATH)
        _, acc = modelo.evaluate(X_test, y_test, verbose=0)
        return modelo, None, acc

    modelo = construir_modelo()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    historial = modelo.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=0
    )

    modelo.save(MODEL_PATH)
    _, acc = modelo.evaluate(X_test, y_test, verbose=0)
    return modelo, historial, acc

# ─────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────
digits_raw, X, y, X_train, X_test, y_train, y_test = cargar_datos()

# ─────────────────────────────────────────────
# Entrenamiento con barra de progreso
# ─────────────────────────────────────────────
with st.spinner("⚙️ Cargando / entrenando el modelo CNN..."):
    modelo, historial, accuracy_test = obtener_modelo(
        X_train, y_train, X_test, y_test
    )

st.success(f"✅ Modelo listo — Precisión en test: **{accuracy_test * 100:.2f}%**")

# ─────────────────────────────────────────────
# Curvas de entrenamiento (solo si se entrenó ahora)
# ─────────────────────────────────────────────
if historial is not None:
    with st.expander("📈 Ver curvas de entrenamiento"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

        ax1.plot(historial.history["accuracy"],    label="Entrenamiento")
        ax1.plot(historial.history["val_accuracy"], label="Validación")
        ax1.set_title("Precisión")
        ax1.set_xlabel("Época")
        ax1.legend()

        ax2.plot(historial.history["loss"],     label="Entrenamiento")
        ax2.plot(historial.history["val_loss"],  label="Validación")
        ax2.set_title("Pérdida")
        ax2.set_xlabel("Época")
        ax2.legend()

        st.pyplot(fig)

st.divider()

# ─────────────────────────────────────────────
# Selección de imagen
# ─────────────────────────────────────────────
st.subheader("🖼️ Selecciona una imagen del dataset")

col1, col2 = st.columns([1, 2])

with col1:
    indice = st.number_input(
        "Índice de la imagen (0 – 1796)",
        min_value=0, max_value=len(X) - 1,
        value=0, step=1
    )

    if st.button("🎲 Imagen aleatoria"):
        indice = int(np.random.randint(0, len(X)))
        st.session_state["indice"] = indice
        st.rerun()

# Recuperar índice de sesión si se usó el botón aleatorio
if "indice" in st.session_state:
    indice = st.session_state["indice"]

# ─────────────────────────────────────────────
# Predicción
# ─────────────────────────────────────────────
imagen = X[indice]                         # (8, 8, 1) normalizada
etiqueta_real = y[indice]

imagen_input = np.expand_dims(imagen, axis=0)   # (1, 8, 8, 1)
probabilidades = modelo.predict(imagen_input, verbose=0)[0]
prediccion = int(np.argmax(probabilidades))
confianza  = float(np.max(probabilidades)) * 100

# ─────────────────────────────────────────────
# Visualización de resultados
# ─────────────────────────────────────────────
with col1:
    fig_img, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(imagen.reshape(IMG_SIZE, IMG_SIZE), cmap="gray_r")
    ax.set_title(f"Imagen #{indice}", fontsize=10)
    ax.axis("off")
    st.pyplot(fig_img)

with col2:
    if prediccion == etiqueta_real:
        st.success(f"### ✅ Predicción: **{prediccion}**")
    else:
        st.error(f"### ❌ Predicción: **{prediccion}** (real: {etiqueta_real})")

    st.metric("Confianza", f"{confianza:.1f}%")
    st.metric("Etiqueta real", etiqueta_real)

# ─────────────────────────────────────────────
# Gráfico de probabilidades
# ─────────────────────────────────────────────
st.subheader("📊 Probabilidades por clase")

fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
colores = ["#2ecc71" if i == etiqueta_real else
           "#e74c3c" if i == prediccion  else
           "#95a5a6"
           for i in range(10)]
bars = ax_bar.bar(range(10), probabilidades * 100, color=colores)
ax_bar.set_xticks(range(10))
ax_bar.set_xlabel("Dígito")
ax_bar.set_ylabel("Probabilidad (%)")
ax_bar.set_ylim(0, 110)
for bar, prob in zip(bars, probabilidades):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{prob*100:.1f}%", ha="center", va="bottom", fontsize=7)

ax_bar.legend(
    handles=[
        plt.Rectangle((0,0),1,1, color="#2ecc71", label="Clase real"),
        plt.Rectangle((0,0),1,1, color="#e74c3c", label="Predicción"),
    ],
    loc="upper right", fontsize=8
)
st.pyplot(fig_bar)

# ─────────────────────────────────────────────
# Galería rápida de ejemplos del dígito predicho
# ─────────────────────────────────────────────
with st.expander(f"🔍 Ver más ejemplos del dígito '{prediccion}' en el dataset"):
    indices_clase = np.where(y == prediccion)[0][:10]
    fig_gal, axes = plt.subplots(2, 5, figsize=(8, 3.5))
    for ax_g, idx in zip(axes.flat, indices_clase):
        ax_g.imshow(X[idx].reshape(IMG_SIZE, IMG_SIZE), cmap="gray_r")
        ax_g.axis("off")
    st.pyplot(fig_gal)

st.caption("Dataset: sklearn.datasets.load_digits | Modelo: CNN con TensorFlow/Keras")
