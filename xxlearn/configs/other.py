import re
from inspect import getsourcelines
from scipy.stats._distn_infrastructure import rv_generic

__all__ = ['Grid', 'Sample', 'Follow', 'Condition', 'Trial', 'Args', '_is_distribution', 'Dist']

def _is_distribution(param):
    if hasattr(param, 'dist') and isinstance(param.dist, rv_generic):
        return True
    return False

class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert key != 'd', "property d is reserved for other purpose"
            self.set(key, value)
    @property
    def d(self):
        return self.__dict__
    def update(self, **kwargs):
        for k,v in kwargs.items():
            self.set(k, v)
    def pop(self,__name):
        return self.__dict__.pop(__name)
    def set(self,__name, __value):
        setattr(self, __name, __value)
    def clone(self):
        from copy import deepcopy
        return deepcopy(self)
    def get(self, __name):
        return self.get_helper(__name, self.d)
    @staticmethod
    def get_helper(_key, _dict):
        if _key in _dict: 
            return _dict[_key]
        for k,v in _dict.items():
            if isinstance(v, dict):
                __key = _key.replace(f'{k}.', '', 1)
                a = Args.get_helper(__key, v)
                if a is not None: return a
        return None
    def __iter__(self):
        yield from self.__dict__


class Grid(list):
    r"""A grid search over a list of values. """
    def __init__(self, values):
        super().__init__(values)
    def __str__(self): 
        return ','.join(map(str, self))
    @property
    def key(self): # [1,2]
        if len(self) > 1:
            return self

class Sample(object):
    def __init__(self, f):
        self.f = f
    def __call__(self):
        return self.f()
    def __str__(self):
        return self.f.__name__

class Follow(object):
    def __init__(self, follow_name, value=None):
        assert isinstance(follow_name, str), "varialbe name to follow should be string"
        self.follow_name = follow_name
        self.value = value
        self.follow_value = None
    def follow(self, other: list):
        if isinstance(other, Follow):
            self.follow_value = other.value
        elif isinstance(other, list):
            self.follow_value = other
    def __str__(self):
        return f'&{self.follow_name}'
    def __call__(self, x):
        if self.value:
            return self.value[self.follow_value.index(x)]
        else:
            return x
    @property
    def key(self):
        return self.value


    
class Condition(object):
    def __init__(self, f):
        assert callable(f)
        self.f = f
    def __call__(self, config):
        return self.f(config)
    def __str__(self):
        return re.search(r'\(lambda x: (.*?)\)',getsourcelines(self.f)[0][0]).group(1)


class Trial(object):
    def __init__(self, items):
        assert isinstance(items, dict), f'dict type expected, got {type(items)}'
        self.all = items
    def __getattr__(self, __name: str):
        if __name == 'all': return self.all
        return self.all.get(__name, None)
    def __contains__(self, __name):
        return __name in self.all
    def __iter__(self):
        for k,v in self.all.items():
            yield (v[0], k, v[1])



class Dist(Sample):

    def __init__(self, param, name=None):
        self.dist = param
        self.is_rv = hasattr(param, 'rvs')
        self.name = name
    
    @property
    def key(self): 
        return self.dist.rvs(2)

    def __call__(self):
        if not self.is_rv: return self.dist
        return self.dist.rvs(1).item()
    
    def __str__(self):
        return self.name

    def rvs(self, *args, **kwargs):
        return self()