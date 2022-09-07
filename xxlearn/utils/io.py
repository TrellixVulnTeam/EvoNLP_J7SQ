from pathlib import Path
import os, re, pickle, cloudpickle, hashlib, yaml, torch
import numpy as np
from typing import Union, Literal, Mapping
from functools import partial
from skorch.utils import is_torch_data_type, to_device, is_pandas_ndframe
from torch.nn.utils.rnn import PackedSequence

__all__ = ['ojoin', 'ofind', 'pickle_load', 'to_tensor', 'pickle_dump', 'CloudpickleWrapper', 'to_numpy', '_is_slicedataset', 'to_device']

def ojoin(a, b):
    return os.path.join(a,b)

def ofind(path, pattern):
    pattern = re.compile(pattern)
    for p,_,file in os.walk(path):
        for each in file:
            if pattern.search(each):
                yield ojoin(path, each)

def _is_slicedataset(X):
    return hasattr(X, 'dataset') and hasattr(X, 'idx') and hasattr(X, 'indices')
                
def to_device(X, device):
    if device is None: 
        return X
    if isinstance(X, Mapping):
        return type(X)({key: to_device(val, device) for key, val in X.items()})
    if isinstance(X, (tuple, list)) and (type(X) != PackedSequence):
        return type(X)(to_device(x, device) for x in X)
    if isinstance(X, torch.distributions.distribution.Distribution):
        return X
    return X.to(device)
                
def to_numpy(X):
    if isinstance(X, np.ndarray):
        return X
    if isinstance(X, dict):
        return {key: to_numpy(val) for key, val in X.items()}
    if is_pandas_ndframe(X):
        return X.values
    if isinstance(X, (tuple, list)):
        return type(X)(to_numpy(x) for x in X)
    if _is_slicedataset(X):
        return np.asarray(X)
    if not is_torch_data_type(X):
        raise TypeError("Cannot convert this data type to a numpy array.")
    if X.is_cuda: X = X.cpu()
    if hasattr(X, 'is_mps') and X.is_mps: X = X.cpu()
    if X.requires_grad: X = X.detach()
    return X.numpy()

# pylint: disable=not-callable
def to_tensor(X, device):
    """@Overwrite skorch function
    """
    def transform(X):
        if X.dtype=='float64': X = X.astype(np.float32)
        elif X.dtype == 'int32': X = X.astype(np.int64)
        return X
    to_tensor_ = partial(to_tensor, device=device)
    if isinstance(X, np.ndarray):
        X = transform(X)
        try:
            return torch.as_tensor(X, device=device)
        except TypeError:
            return X
    if is_torch_data_type(X):
        return to_device(X, device)
    if hasattr(X, 'convert_to_tensors'):
        return X.convert_to_tensors('pt')   # huggingface API
    if isinstance(X, dict):
        return {key: to_tensor_(val) for key, val in X.items()}
    if isinstance(X, (list, tuple)):
        return [to_tensor_(x) for x in X]
    return X

def hashcode(s: Union[Literal['str'], Literal['dict']]):
    if isinstance(s, dict):
        s = pickle.dumps(s)
    else:
        s = s.encode('utf-8')
    return hashlib.md5(s).hexdigest()

def pickle_load(f, ext='.pkl', encode="ASCII"):
    r"""Read a pickled data from a file. 

    Args:
        f (str/Path): file path
    """
    if isinstance(f, Path):
        f = f.as_posix() 
        f += [ext,''][f[-4:]==ext]

    with open(f, 'rb') as file:
        return cloudpickle.load(file, encoding=encode)


def pickle_dump(obj, f, ext='.pkl'):
    r"""Serialize an object using pickling and save in a file. 
    
    .. note::
    
        It uses cloudpickle instead of pickle to support lambda
        function and multiprocessing. By default, the highest
        protocol is used. 
        
    .. note::
    
        Except for pure array object, it is not recommended to use
        ``np.save`` because it is often much slower. 
    
    Args:
        obj (object): a serializable object
        f (str/Path): file path
        ext (str, optional): file extension. Default: .pkl
    """
    if isinstance(f, Path):
        f = f.as_posix()
        f += [ext,''][f[-4:]==ext]
    
    with open(f, 'wb') as file:
        return cloudpickle.dump(obj=obj, file=file, protocol=pickle.HIGHEST_PROTOCOL)


def yaml_load(f):
    r"""Read the data from a YAML file. 

    Args:
        f (str/Path): file path
    """
    if isinstance(f, Path):
        f = f.as_posix()
    
    with open(f, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def yaml_dump(obj, f, ext='.yml'):
    r"""Serialize a Python object using YAML and save in a file. 
    
    .. note::
    
        YAML is recommended to use for a small dictionary and it is super
        human-readable. e.g. configuration settings. For saving experiment
        metrics, it is better to use :func:`pickle_dump`.
        
    .. note::
    
        Except for pure array object, it is not recommended to use
        ``np.load`` because it is often much slower. 
        
    Args:
        obj (object): a serializable object
        f (str/Path): file path
        ext (str, optional): file extension. Default: .yml
        
    """
    if isinstance(f, Path):
        f = f.as_posix()
    with open(f+ext, 'w') as file:
        return yaml.dump(obj, file, sort_keys=False)

    
class CloudpickleWrapper(object):
    r"""Uses cloudpickle to serialize contents (multiprocessing uses pickle by default)
    
    This is useful when passing lambda definition through Process arguments.
    """
    def __init__(self, x):
        self.x = x
        
    def __call__(self, *args, **kwargs):
        return self.x(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.x, name)
    
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
                
if __name__ == '__main__':
    pass

