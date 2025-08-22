# This class is to create a Neural Network Library Built upon TensorFlow

import numpy as np
from keras.src.layers import Input
from tensorflow.keras import layers, models
import tensorflow as tf
from TensorFlowService import STBlock as stb
from ModelStorageService import ModelStorageService as st
import json
import os
import joblib
import DataPreparation_graph as dt
from sklearn.preprocessing import StandardScaler


class ModelService:

    def __init__(self, train_set, test_set, train_labels, test_labels, validation_set, validation_labels):
        self.train_set = train_set
        self.test_set = test_set
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.validation_set = validation_set
        self.validation_labels = validation_labels
        pass

    # Utils-like function to normalize the Adjacency Matrix
    def normalize_adj_random_walk(self, A):
        A_hat = A + np.eye(A.shape[0], dtype=A.dtype)
        d = A_hat.sum(axis=1)
        D_inv = np.diag(1.0 / np.maximum(d, 1e-8))
        return D_inv @ A_hat

    # Utils-like function to compute Model Supports
    def compute_supports(self, A_norm, max_power=2):
        supports = [A_norm]
        Xp = A_norm.copy()
        for _ in range(2, max_power):
            Xp = Xp @ A_norm
            supports.append(Xp)
        return supports

    def build_graph_wavenet(self, N, F_in, W, H, A,
                            channels_t=32, channels_s=32,
                            n_blocks=3, dilations=(1, 2, 4),
                            kernel_size=2):
        """
        N: number of nodes
        F_in: feature per node
        W: window_size (input)
        H: horizon (output)
        A: adjacency (numpy matrix 2x2) according to the split passed to the model
        """
        # Normalize + create supports
        A_norm = self.normalize_adj_random_walk(A)
        supports = self.compute_supports(A_norm, max_power=2)  # [A, A^2]

        # Input: (B, N, F, W) -> Change to a (B, W, N, C)
        X_in = Input(shape=(N, F_in, W), name="X")
        x = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 3, 1, 2]))(X_in)  # (B, W, N, F)

        # Initial Channels Projection
        x = layers.Conv2D(filters=channels_t, kernel_size=(1, 1), padding="same")(x)  # (B, W, N, C)

        # Pile the ST Blocks
        skips = []
        for b, d in enumerate(dilations[:n_blocks]):
            st = stb.STBlock(channels_t=channels_t, channels_s=channels_s, supports=supports,
                         kernel_size=kernel_size, dilation=d, name=f"stblock_{b}")
            x, skip = st(x)
            skips.append(skip)

        # Skip connection aggregation
        s = layers.Add()(skips)
        s = layers.Activation('relu')(s)
        s = layers.Conv2D(filters=channels_t, kernel_size=(1, 1), activation='relu', padding="same")(s)

        # temporal head to create H steps
        # (B, W, N, C) -> time-Conv1D inside a Conv2D with kernel (k,1)
        # We use here a 1x1 to directly map the channels F_in*H on time dimensions
        # Compress on time and then expand at H
        s = layers.Conv2D(filters=channels_t, kernel_size=(1, 1), activation='relu', padding="same")(s)
        # Map directly an F_in * H on time dimension with a 1x1+reshape
        out = layers.Conv2D(filters=F_in * H, kernel_size=(1, 1), padding="same")(s)  # (B, W, N, F*H)

        # Take the last row as causal "decision" + remodel it at (B, N, F, H)
        def take_last_timestep(t):
            # t: (B, W, N, F*H)
            last = t[:, -1, :, :]  # (B, N, F*H)
            return tf.reshape(last, (-1, N, F_in, H))  # (B, N, F, H)

        Y_out = layers.Lambda(take_last_timestep, name="forecast")(out)

        model = models.Model(inputs=X_in, outputs=Y_out, name="GraphWaveNet_Minimal")

        return model

    def WaveNetTimeSpaceModel (self, adj_matrix_train, training_epochs):

        # Extract Dimensions from input set
        N_train = self.train_set.shape[1]
        F_in = self.train_set.shape[2]  # ex. 1
        W = self.train_set.shape[3]  # ex. 24
        H = self.train_labels.shape[3]  # ex. 96

        model = self.build_graph_wavenet(
            N=N_train, F_in=F_in, W=W, H=H, A=adj_matrix_train,
            channels_t=32, channels_s=32,
            n_blocks=3, dilations=(1, 2, 4), kernel_size=2
        )

        model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
                      loss="mse",
                      metrics=[tf.keras.metrics.MAE])

        # Training: A_train is "frozen" implicitly inside the training algorithm
        history = model.fit(
            self.train_set, self.train_labels,
            validation_data=(self.validation_set, self.validation_labels),
            epochs=training_epochs,
            batch_size=4,
            verbose=1
        )

        # Evaluation on the test set
        test_metrics = model.evaluate(self.test_set, self.test_labels, batch_size=4, verbose=1)

        return test_metrics






