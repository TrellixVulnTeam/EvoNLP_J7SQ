from .mnist import *
import sys
import numpy as np

__all__ = [x for x in dir() if not x.startswith('__')]

def make_data(data_name, subset_ratio=1.0):
    data_module = getattr(sys.modules[__name__], data_name)
    data, target = data_module().load_data()
    indices = np.random.permutation(len(data))
    N = int(len(data) * subset_ratio)
    return data[indices][:N], target[indices][:N]