from pathlib import Path
import os, re, pickle, cloudpickle, hashlib, yaml
from typing import Union, Literal

__all__ = ['ojoin', 'ofind', 'pickle_load', 'pickle_dump', 'CloudpickleWrapper']

def ojoin(a, b):
    return os.path.join(a,b)

def ofind(path, pattern):
    pattern = re.compile(pattern)
    for p,_,file in os.walk(path):
        for each in file:
            if pattern.search(each):
                yield ojoin(path, each)



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

