# Class for diffusion Graph Convolutions

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class DiffusionGraphConv(layers.Layer):
    """
    Apply con on a graph with different supports (es. A, A^2) time-wise.
    Input:  (B, W, N, C_in)
    Output: (B, W, N, C_out)
    """
    def __init__(self, supports, channels_out, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.channels_out = channels_out
        self.use_bias = use_bias
        # Freeze supports
        self.supports = [tf.constant(S.astype(np.float32)) for S in supports]

    def build(self, input_shape):
        C_in = int(input_shape[-1])
        S = len(self.supports)
        # Create one param for each one of the supports
        self.theta = self.add_weight(
            shape=(S, C_in, self.channels_out),
            initializer="glorot_uniform",
            trainable=True,
            name="theta"
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.channels_out,),
                initializer="zeros",
                trainable=True,
                name="bias"
            )
        else:
            self.b = None

    def call(self, x):
        # x: (B, W, N, C_in) → (B*W, N, C_in)
        B, W, N, C_in = tf.unstack(tf.shape(x))
        x2 = tf.reshape(x, (B*W, N, C_in))

        # For each one of the supports: (B*W, N, C_in) -> (B*W, N, C_out)
        outs = []
        for s, S in enumerate(self.supports):
            # (B*W, N, C_in) ← (B*W, N, N) @ (B*W, N, C_in) requires broadcast:
            # use einsum: (n,n) x (b,n,c) -> (b,n,c)
            x_s = tf.einsum('ij,bjc->bic', S, x2)     # mix between nodes
            out_s = tf.tensordot(x_s, self.theta[s], axes=[[2],[0]])  # (B*W, N, C_out)
            outs.append(out_s)

        h = tf.add_n(outs)
        if self.b is not None:
            h = tf.nn.bias_add(h, self.b)
        # back to (B, W, N, C_out)
        h = tf.reshape(h, (B, W, N, self.channels_out))
        return h