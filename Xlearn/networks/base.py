from typing import Iterable
import torch.nn as nn
from Xlearn.utils.layers import tween, flatten

__all__ = ['CNN1D']

class CNN1D(nn.Sequential):

    """
    >>> n_channels = [3, 16, 16, 32, 32, 64, 64, 128, 128]
    >>> channel_sizes = [5, 5, 4, 4, 4, 4, 4, 4, 5]
    >>> model = CNN1D(n_channels, channel_sizes)
    >>> input = torch.randn(10, 3, 4000)
    >>> output = model(input)  # shape (10, 128, 12)
    """
    
    def __init__(self, n_channels=[24,12], channel_sizes=[5,2], pool_sizes=2, learn_batch=False):
        if not isinstance(n_channels, Iterable):
            n_channels = [n_channels] * (len(channel_sizes)+1)
        conv = [
            (
                nn.Conv1d( in_channels=a, out_channels=b, kernel_size=k), 
                nn.BatchNorm1d(num_features=b, affine=learn_batch)
            )   for a, b, k in zip(n_channels[:-1], n_channels[1:], channel_sizes)
        ]
        fixed = [
            nn.LeakyReLU(inplace=True), nn.MaxPool1d(kernel_size=pool_sizes)
        ]
        super().__init__(*flatten(tween(conv, fixed, add_last=True)))

        




