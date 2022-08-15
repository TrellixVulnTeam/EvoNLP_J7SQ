import re
from itertools import product
from inspect import getsourcelines
from typing import Tuple

__all__ = ['Grid', 'Book', 'Sample', 'Follow', 'Case', 'Condition', 'Trial', 'Const', 'Args', 'Config']

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

class Book(dict):
    def __init__(self, values):
        super().__init__(values)
    def __str__(self): 
        return ','.join(map(str, self)) #TODO: unimplemented
    @property
    def key(self): # {'a': [1, 2], 'd': ['a', 'b']}
        return Config(self).obtain_key()


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

class Case(object):
    def __init__(self, follow_name, value_d: dict = None):
        self.follow_name = follow_name
        if not 'default' in value_d: 
            value_d['default'] = {}
        if isinstance(value_d['default'], dict):
            self.rtn_dict = True
        else:
            self.rtn_dict = False
        self.value_d = value_d
    def __call__(self, key, default_d={}):
        if self.rtn_dict:
            return {**default_d, **self.value_d.get(key, {})}
        else:
            return self.value_d.get(key, self.value_d['default'])
    def __str__(self):
        return f'&{self.follow_name}'
    @property
    def key(self):
        return Config(self.value_d['default']).obtain_key()
    
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

class Const:
    """
    Define constant variable used in this module
    """
    EXP = 'exp_name'
    MDL = 'model_name'
    SRN = 'search_name'
    KEY = 'optim_metric'
    DIR = 'optim_direction'
    NTrial = 'number_of_trials'
    DAT = 'data_name'
    EMB = 'embed_name'
    OUTPUT = 'pred'
    TARGET = 'target'


class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert key != 'd', "property d is reserved for other purpose"
            self.set(key, value)
    @property
    def d(self):
        return self.__dict__
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

class Config(object):
    r"""Defines a set of configurations for the experiment. 
    
    The configuration includes the following possible items:
    
    * Hyperparameters: learning rate, batch size etc.
    
    * Experiment settings: training iterations, logging directory, environment name etc.
    
    All items are stored in a dictionary. It is a good practice to semantically name each item
    e.g. `network.lr` indicates the learning rate of the neural network. 
    
    For hyperparameter search, we support both grid search (:class:`Grid`) and random search (:class:`Sample`).
    
    Call :meth:`make_configs` to generate a list of all configurations, each is assigned
    with a unique ID. 
    
    note::
    
        For random search over small positive float e.g. learning rate, it is recommended to
        use log-uniform distribution, i.e.
        .. math::
            \text{logU}(a, b) \sim \exp(U(\log(a), \log(b)))
        
        An example: `np.exp(np.random.uniform(low=np.log(low), high=np.log(high)))`
            
        Because direct uniform sampling is very `numerically unstable`_.
        
    .. warning::
    
        The random seeds should not be set here. Instead, it should be handled by
        :class:`BaseExperimentMaster` and :class:`BaseExperimentWorker`.
    
    Example::
    
        >>> config = Config({'log.dir': 'some path', 'network.lr': Grid([1e-3, 5e-3]), 'env.id': Grid(['CartPole-v1', 'Ant-v2'])}, num_sample=1, keep_dict_order=False)
        >>> import pandas as pd
        >>> print(pd.DataFrame(config.make_configs()))
               ID       env.id    log.dir  network.lr
            0   0  CartPole-v1  some path       0.001
            1   1       Ant-v2  some path       0.001
            2   2  CartPole-v1  some path       0.005
            3   3       Ant-v2  some path       0.005
    
    Args:
        items (dict): a dictionary of all configuration items. 
        num_sample (int): number of samples for random configuration items. 
            If grid search is also provided, then the grid will be repeated :attr:`num_sample`
            of times. 
        keep_dict_order (bool): if ``True``, then each generated configuration has the same
            key ordering with :attr:`items`. 
            
    .. _numerically unstable:
            http://cs231n.github.io/neural-networks-3/#hyper
    """
    def __init__(self, items, num_sample=1):
        #TODO: right now dictionary argument isn't supporting very well
        #TODO: they do not appear in the obtain_key output
        if not isinstance(items, dict):
            items = {}
        self.item_d = items
        self.param_d = {}
        self.num_sample = num_sample

    def obtain_key(self, tune=[]):
        #TODO: right now, Sample parameter isn't included !!
        new_dict = {k: v for k,v in self.item_d.items() if k not in tune}
        param_d = {}

        for key,param in new_dict.items():
            if isinstance(param, Grid):
                param_d[key] = param.key
        for key,param in new_dict.items():
            if isinstance(param, Follow) and param.follow_name in param_d:
                param_d[key] = param.key or param_d[param.follow_name]
        for key,param in new_dict.items():
            if isinstance(param, Case):
                for k,v in param.key.items():
                    param_d[f'{key}.{k}'] = v
            elif isinstance(param, (dict, Book)):
                for k,v in Config(param).obtain_key().items():
                    param_d[f'{key}.{k}'] = v

        for k in param_d.copy(): #delete parameter with only one choice
            if param_d[k] is None or len(param_d[k]) <= 1:
                param_d.pop(k)
        self.item_d, self.param_d = new_dict, {**new_dict, **param_d}
        return param_d

    def __getitem__(self, __name):
        return self.item_d.get(__name, None)

    def __contains__(self, __name):
        return __name in self.item_d

    def __iter__(self):
        yield from self.param_d
    
    def make_configs(self, new_id=True, **kwargs):
        r"""Generate a list of all possible combinations of configurations, including
        grid search and random search. 
        
        Returns:
            list: a list of all possible configurations
        """
        keys_grid = {}
        keys_fixed, keys_sample = [[] for _ in range(2)] 
        self.item_d = {**self.item_d, **kwargs}
        for key, param in self.item_d.items():
            if isinstance(param, (dict, Book)):
                vlist = Config(param).make_configs(new_id=False)
                keys_grid[key] = vlist
            elif isinstance(param, Case):
                vlist = Config(param.value_d['default']).make_configs(new_id=False)
                keys_grid[key] = [(param, v) for v in vlist]
            elif isinstance(param, Grid):
                keys_grid[key] = param
            elif isinstance(param, Sample):
                keys_sample.append(key)
            elif isinstance(param, Follow):
                param.follow(self.item_d[param.follow_name])
                keys_fixed.append(key)
            else:
                keys_fixed.append(key)
                
        grid_product = list(
            dict(zip(keys_grid.keys(), values)) for values in product(*keys_grid.values())
        )
        list_rtn = []
        fixed_d = {key: self.item_d[key] for key in keys_fixed}
        for i in range(len(grid_product)*self.num_sample):
            d = {'ID': i} if new_id else {}
            d = {**d, **fixed_d, **grid_product[i%len(grid_product)]}
            for key in keys_sample:
                d[key] = self.item_d[key]()
            for key, value in d.items():
                if isinstance(value, Condition):
                    d[key] = value(d)
                elif isinstance(value, Follow):
                    d[key] = value(d[value.follow_name]) 
                elif isinstance(value, Tuple):
                    fn, default_d = value
                    d[key] = fn(d[fn.follow_name], default_d)
            list_rtn.append(d)
        return list_rtn