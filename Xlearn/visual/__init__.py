import torch as T
import torch.nn as nn
from .analysis import *

class RF:
    """Receptive Field
    Args:
        model(callable)     network to estimate receptive field from
        input_shape(tuple)  can be (C,W) or (C,W,H)
        n_sample(int)       number of sample to estimate gradient from

    >>> model = CNN1D()
    """
    def __init__(self, model, input_shape, n_sample=10):
        pass

    

def get_name(module):
    return module.__class__.__name__

if __name__ == '__main__':
    from doctest import testmod
    from Xlearn.networks import CNN1D
    import torch
    n_channels = [3, 16, 16, 32, 32, 64, 64, 128, 128]
    channel_sizes = [5, 5, 4, 4, 4, 4, 4, 4, 5]
    model = CNN1D(n_channels, channel_sizes)
    input = torch.randn(10, 3, 4000)