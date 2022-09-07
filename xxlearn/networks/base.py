from typing import Iterable
import torch
import torch.nn as nn
from xxlearn.utils.layers import tween, flatten

__all__ = ['CNN1D']

class CNN1D(nn.Sequential):

    """
    >>> n_channels = [3, 16, 16, 32, 64, 64, 128]
    >>> channel_sizes = [5, 5, 4, 4, 4, 4, 4, 4, 5]
    >>> model = CNN1D(n_channels, channel_sizes)
    >>> input = torch.randn(10, 3, 4000)
    >>> output = model(input)
    >>> output.shape
    torch.Size([10, 128, 12])
    """
    
    def __init__(self, n_channels=[24,12], channel_sizes=[5,2], pool_sizes=2, learn_batch=False, deep=None, last_pool=True, dropout=0):
        if not isinstance(n_channels, Iterable):
            n_channels = [n_channels] * (len(channel_sizes)+1)
        deep = deep or len(channel_sizes)
        conv = []
        for a, b, k in zip(n_channels[:-1], n_channels[1:], channel_sizes):
            conv.append((
                nn.Conv1d( in_channels=a, out_channels=b, kernel_size=k), 
                nn.Conv1d( in_channels=b, out_channels=b, kernel_size=k) if deep>0 else [], 
                nn.BatchNorm1d(num_features=b, affine=learn_batch)
            ))
            deep -= 1
        fixed = [
            nn.LeakyReLU(inplace=True), nn.MaxPool1d(kernel_size=pool_sizes)
        ]
        if dropout > 0:
            fixed.insert(1, nn.Dropout1d(p=dropout))
        self.in_channel = n_channels[0]
        super().__init__(*flatten(tween(conv, fixed, add_last=last_pool)))

    def summary(self, input_dim=300):
        x = torch.rand(1,self.in_channel,input_dim)
        for layer in self:
            x_ = x.size(-1)
            x = layer(x)
            if x_ == x.size(-1): continue
            print('{0:<80}{1:>10}->{2}'.format(str(layer), x_, x.size(-1)))



if __name__ == '__main__':
    import doctest
    doctest.testmod()


