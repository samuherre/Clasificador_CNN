import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import os

MODEL_PATH = "mnist_cnn.h5"

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizar
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape para CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model():
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    model.save(MODEL_PATH)
    print("Modelo guardado")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model()
