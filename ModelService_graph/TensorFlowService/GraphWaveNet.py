# Class to handle GraphWaveNet: it must be separated from the data class (in ModelService)

import numpy as np
from keras.src.layers import Input
from tensorflow.keras import layers, models
import tensorflow as tf
from ModelService_graph.TensorFlowService import STBlock as stb

class GraphWaveNet:
    def __init__(self, N, F_in, W, H, A, channels_t=32, channels_s=32, n_blocks=3, dilations=(1, 2, 4), kernel_size=2):
        self.N = N
        self.F_in = F_in
        self.W = W
        self.H = H
        self.A = A
        self.channels_t = channels_t
        self.channels_s = channels_s
        self.n_blocks = n_blocks
        self.dilations = dilations
        self.kernel_size = kernel_size
        pass

    # Normalize Random Walk function
    def normalize_adj_random_walk(self, A):
        A_hat = A + np.eye(A.shape[0], dtype=A.dtype)
        d = A_hat.sum(axis=1)
        D_inv = np.diag(1.0 / np.maximum(d, 1e-8))
        return D_inv @ A_hat

    # Compute supports function
    def compute_supports(self, A_norm, max_power=2):
        supports = [A_norm]
        Xp = A_norm.copy()
        for _ in range(2, max_power):
            Xp = Xp @ A_norm
            supports.append(Xp)
        return supports

    # Class that builds the model
    def build_graph_wavenet(self):
        """
        N: number of nodes
        F_in: feature per node
        W: window_size (input)
        H: horizon (output)
        A: adjacency (numpy matrix 2x2) according to the split passed to the model
        """
        # Normalize + create supports
        A_norm = self.normalize_adj_random_walk(self.A)
        supports = self.compute_supports(A_norm, max_power=2)  # [A, A^2]

        # Input: (B, N, F, W) -> Change to a (B, W, N, C)
        X_in = Input(shape=(self.N, self.F_in, self.W), name="X")
        x = layers.Lambda(lambda t: tf.transpose(t, perm=[0, 3, 1, 2]))(X_in)  # (B, W, N, F)

        # Initial Channels Projection
        x = layers.Conv2D(filters=self.channels_t, kernel_size=(1, 1), padding="same")(x)  # (B, W, N, C)

        # Pile the ST Blocks
        skips = []
        for b, d in enumerate(self.dilations[:self.n_blocks]):
            st = stb.STBlock(channels_t=self.channels_t, channels_s=self.channels_s, supports=supports,
                         kernel_size=self.kernel_size, dilation=d, name=f"stblock_{b}")
            x, skip = st(x)
            skips.append(skip)

        # Skip connection aggregation
        s = layers.Add()(skips)
        s = layers.Activation('relu')(s)
        s = layers.Conv2D(filters=self.channels_t, kernel_size=(1, 1), activation='relu', padding="same")(s)

        # temporal head to create H steps
        # (B, W, N, C) -> time-Conv1D inside a Conv2D with kernel (k,1)
        # We use here a 1x1 to directly map the channels F_in*H on time dimensions
        # Compress on time and then expand at H
        s = layers.Conv2D(filters=self.channels_t, kernel_size=(1, 1), activation='relu', padding="same")(s)
        # Map directly an F_in * H on time dimension with a 1x1+reshape
        out = layers.Conv2D(filters=self.F_in * self.H, kernel_size=(1, 1), padding="same")(s)  # (B, W, N, F*H)

        # Take the last row as causal "decision" + remodel it at (B, N, F, H)
        def take_last_timestep(t):
            # t: (B, W, N, F*H)
            last = t[:, -1, :, :]  # (B, N, F*H)
            return tf.reshape(last, (-1, self.N, self.F_in, self.H))  # (B, N, F, H)

        Y_out = layers.Lambda(take_last_timestep, name="forecast")(out)
        model = models.Model(inputs=X_in, outputs=Y_out, name="GraphWaveNet_Minimal")

        return model
