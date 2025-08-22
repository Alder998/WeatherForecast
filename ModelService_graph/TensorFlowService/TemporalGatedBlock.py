# Implementation class for temporal Gated Blocks

import tensorflow as tf
from tensorflow.keras import layers

class TemporalGatedBlock(layers.Layer):
    """
    TCN with conv1D causal dilated + gated activation + residual/skip.
    It works node-by-node, therefore we need to remodel (B, W, N, C) -> (B*N, W, C) -> Conv1D -> back.
    """
    def __init__(self, channels, kernel_size=2, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.conv_f = layers.Conv1D(
            filters=channels, kernel_size=kernel_size, padding="causal",
            dilation_rate=dilation_rate
        )
        self.conv_g = layers.Conv1D(
            filters=channels, kernel_size=kernel_size, padding="causal",
            dilation_rate=dilation_rate
        )
        self.res_proj = layers.Conv1D(filters=channels, kernel_size=1)
        self.skip_proj = layers.Conv1D(filters=channels, kernel_size=1)

    def call(self, x):
        # x: (B, W, N, C)
        B, W, N, C = tf.unstack(tf.shape(x))
        x_ = tf.reshape(x, (B * N, W, C))  # node-by-node

        f = self.conv_f(x_)
        g = self.conv_g(x_)
        h = tf.nn.tanh(f) * tf.nn.sigmoid(g)

        # Residual Projection and skip
        residual = self.res_proj(h)         # (B*N, W, C)
        skip = self.skip_proj(h)            # (B*N, W, C)

        # Align temporal Prediction with input (Conv1D causal keeps W)
        out = x_ + residual                 # residual connection

        # back to (B, W, N, C)
        out = tf.reshape(out, (B, W, N, C))
        skip = tf.reshape(skip, (B, W, N, C))
        return out, skip