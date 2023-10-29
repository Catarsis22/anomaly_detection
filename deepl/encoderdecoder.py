import numpy as np
import tensorflow as tf
from tensorflow import keras


class AutoEncoder():
    def __init__(self, layer_activation='relu', output_activation='linear',
                 optimizer='adam', loss='mean_squared_error'):
        self.layer_activation = layer_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.model = None

    def define_model(self):
        # Build a simple autoencoder model
        self.model = keras.Sequential([
            keras.layers.Input(shape=(12,)),
            keras.layers.Dense(8, activation=self.layer_activation),
            keras.layers.Dense(4, activation=self.layer_activation),
            keras.layers.Dense(8, activation=self.layer_activation),
            keras.layers.Dense(12, activation=self.output_activation)  # Output layer with linear activation
        ])

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, x_train, x_test, epochs):
        # Train the autoencoder
        self.model.fit(x_train, x_train, epochs=epochs, batch_size=32, validation_data=(x_test, x_test))

    def anomaly_detection(self, df, threshold):
        # Use the trained autoencoder to detect anomalies
        reconstructions = self.model.predict(df)
        mse = np.mean(np.power(df - reconstructions, 2), axis=1)  # Mean Squared Error

        # Define a threshold for anomaly detection (adjust as needed)
        threshold = threshold

        # Identify anomalies
        anomalies = df[mse > threshold]

        return anomalies.index.values

    def load_ae_model(self, path):
        self.model = keras.models.load_model(path)

    def save_ae_model(self, path):
        self.model.save(path)
