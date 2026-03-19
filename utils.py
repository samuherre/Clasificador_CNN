import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "mnist_cnn.h5"

def load_trained_model():
    return load_model(MODEL_PATH)

def predict_digit(model, image):
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence
