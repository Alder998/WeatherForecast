# Class to implement the structure temporal-spacial-temporal
import numpy as np
from tensorflow.keras import layers
from .DiffusionGraphConv import DiffusionGraphConv
from .TemporalGatedBlock import TemporalGatedBlock

class STBlock(layers.Layer):
    def __init__(self, channels_t, channels_s, has_supports=False, supports=None, kernel_size=2, dilation=1, **kwargs):
        super().__init__(**kwargs)

        # define primary params first
        self.channels_t = channels_t
        self.channels_s = channels_s
        self.supports = supports
        # Boolean value for supports saving
        self.has_supports = supports is not None
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.temp1 = TemporalGatedBlock(channels=channels_t, kernel_size=kernel_size, dilation_rate=dilation)
        self.gconv = DiffusionGraphConv(supports=supports, channels_out=channels_s)
        self.temp2 = TemporalGatedBlock(channels=channels_t, kernel_size=kernel_size, dilation_rate=1)
        self.bn = layers.BatchNormalization()

        # Set supports for save only if it is an array
        if isinstance(supports, (np.ndarray, list, tuple)):
            self.gconv = DiffusionGraphConv(supports=supports, channels_out=channels_s)
        else:
            self.gconv = None  # set placeholder

    def set_supports(self, supports):
        # Re-Initialize after loading
        self.supports = supports
        self.gconv = DiffusionGraphConv(supports=supports, channels_out=self.channels_s)
        self.has_supports = True

    def call(self, x):
        # x: (B, W, N, C)
        h, skip1 = self.temp1(x)                 # temporal 1
        h = self.gconv(h)                        # spatial (graph)
        h, skip2 = self.temp2(h)                 # temporal 2
        h = self.bn(h)
        skip = skip1 + skip2
        return h, skip

    # get_config to save layer within the model
    def get_config(self):
        config = super().get_config()
        config.update({
            "channels_t": self.channels_t,
            "channels_s": self.channels_s,
            "kernel_size": self.kernel_size,
            "dilation": self.dilation,
            "has_supports": self.has_supports,
        })
        return config


