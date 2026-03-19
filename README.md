# 🔢 Reconocimiento de Dígitos con CNN

Aplicación interactiva desarrollada con **Streamlit** y **TensorFlow/Keras** que usa una **Red Neuronal Convolucional (CNN)** para reconocer dígitos del dataset `sklearn.datasets.load_digits`.

---

## 📋 Descripción

El usuario selecciona (por índice o aleatoriamente) una imagen del dataset **Digits de sklearn** (imágenes de 8×8 píxeles, dígitos del 0 al 9) y el modelo CNN predice a qué dígito corresponde, mostrando:

- La imagen seleccionada
- La predicción y la etiqueta real
- El porcentaje de confianza
- Un gráfico de barras con las probabilidades de cada clase
- Curvas de entrenamiento (precisión y pérdida)
- Galería de ejemplos del dígito predicho

---

## 🧠 Arquitectura del Modelo CNN

```
Input (8, 8, 1)
│
├── Conv2D(32, 3×3, relu) + BatchNorm + MaxPool(2×2) + Dropout(0.25)
├── Conv2D(64, 3×3, relu) + BatchNorm + Dropout(0.25)
├── Flatten
├── Dense(128, relu) + BatchNorm + Dropout(0.5)
└── Dense(10, softmax)  ← 10 clases (0–9)
```

- **Optimizador:** Adam  
- **Función de pérdida:** Sparse Categorical Crossentropy  
- **Early Stopping:** paciencia de 5 épocas  
- **Precisión típica en test:** ~98–99%

---

## 🚀 Instalación y ejecución local

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

La app abrirá automáticamente en `http://localhost:8501`.

---

## 📁 Estructura del proyecto

```
📦 tu-repo/
├── app.py                  # Aplicación principal Streamlit
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Este archivo
└── cnn_digits_model.keras  # Modelo entrenado (se genera automáticamente)
```

> **Nota:** el archivo `cnn_digits_model.keras` se genera la primera vez que ejecutas la app. Puedes agregarlo a `.gitignore` si no quieres subir el modelo al repositorio.

---

## 🌐 Despliegue en Streamlit Community Cloud

1. Sube el proyecto a GitHub.
2. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
3. Selecciona el repositorio, la rama y el archivo `app.py`.
4. Haz clic en **Deploy** y espera unos minutos.

---

## 🛠️ Tecnologías usadas

| Tecnología | Uso |
|---|---|
| Python 3.10+ | Lenguaje principal |
| TensorFlow / Keras | Construcción y entrenamiento del CNN |
| scikit-learn | Dataset Digits y split de datos |
| Streamlit | Interfaz web interactiva |
| Matplotlib | Visualizaciones y gráficos |
| NumPy | Operaciones con arreglos |

---

## 📚 Dataset

Se usa `sklearn.datasets.load_digits`:
- **1 797 imágenes** de dígitos del 0 al 9
- Resolución: **8×8 píxeles** (valores entre 0 y 16)
- **10 clases** balanceadas (~180 muestras por clase)

---

## 👨‍💻 Autor

Ejercicio académico — Materia de Inteligencia Artificial
