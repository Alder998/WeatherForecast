# Class to implement the structure temporal-spacial-temporal

from tensorflow.keras import layers
from .DiffusionGraphConv import DiffusionGraphConv
from .TemporalGatedBlock import TemporalGatedBlock

class STBlock(layers.Layer):
    def __init__(self, channels_t, channels_s, supports, kernel_size=2, dilation=1, **kwargs):
        super().__init__(**kwargs)
        self.temp1 = TemporalGatedBlock(channels=channels_t, kernel_size=kernel_size, dilation_rate=dilation)
        self.gconv = DiffusionGraphConv(supports=supports, channels_out=channels_s)
        self.temp2 = TemporalGatedBlock(channels=channels_t, kernel_size=kernel_size, dilation_rate=1)
        self.bn = layers.BatchNormalization()

    def call(self, x):
        # x: (B, W, N, C)
        h, skip1 = self.temp1(x)                 # temporal 1
        h = self.gconv(h)                        # spatial (graph)
        h, skip2 = self.temp2(h)                 # temporal 2
        h = self.bn(h)
        skip = skip1 + skip2
        return h, skip