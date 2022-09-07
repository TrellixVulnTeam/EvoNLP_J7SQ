from itertools import product
from typing import Dict, Tuple
import random
from .other import *
from .sampler import *

__all__ = ['Config', 'Case', 'Book', 'Args', 'Grid']

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

class Book(dict):
    def __init__(self, values):
        super().__init__(values)
    def __str__(self): 
        return ','.join(map(str, self)) #TODO: unimplemented
    @property
    def key(self): # {'a': [1, 2], 'd': ['a', 'b']}
        return Config(self).obtain_key()   

class Config:
    r"""Defines a set of configurations for the experiment. 
    The configuration includes the following possible items:
    * Hyperparameters: learning rate, batch size etc.
    * Experiment settings: training iterations, logging directory, environment name etc.
    All items are stored in a dictionary. It is a good practice to semantically name each item
    e.g. `network.lr` indicates the learning rate of the neural network. 
    For hyperparameter search, we support both grid search (:class:`Grid`) and random search (:class:`Sample`).
    Call :meth:`make_configs` to generate a list of all configurations, each is assigned
    with a unique ID. 
    
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
        n_iter (int): number of samples for random configuration items. 
        keep_dict_order (bool): if ``True``, then each generated configuration has the same key ordering with :attr:`items`. 
    """

    def __init__(self, items, n_iter=1):
        if not isinstance(items, dict):
            items = {}
        self.item_d = items
        self.param_d = {}
        self.n_iter = n_iter
    
    def obtain_key(self):
        param_d = {}
        for key,v in self.item_d.items():
            if isinstance(v, Grid):
                param_d[key] = v.key
            elif isinstance(v, Sampler) and v.key:
                for kk, vv in v.retrieve_from_lst():
                    param_d[f'{key}.{kk}'] = vv
        for k in param_d.copy():        #delete parameter with only one choice
            if param_d[k] is None or len(param_d[k]) <= 1:
                param_d.pop(k)
        return param_d

    def __getitem__(self, __name):
        return self.item_d.get(__name, None)

    def __contains__(self, __name):
        return __name in self.item_d

    def __iter__(self):
        yield from self.param_d

    def make_configs(self, **kwargs):
        r"""Generate a list of all possible combinations of configurations, including
        grid search and random search. 
        
        Returns:
            list: a list of all possible configurations
        """
        self.item_d = {**self.item_d, **kwargs}
        keys_fixed, keys_grid, keys_sample = [], [], []
        for key, x in self.item_d.items():
            if isinstance(x, Grid):     keys_grid.append(key)
            elif isinstance(x, Sample): keys_sample.append(key)
            elif isinstance(x, Dict):   continue
            else:                       keys_fixed.append(key)

        product_grid = list(product(*[self.item_d[key] for key in keys_grid]))
        total_generation_times = len(product_grid)*self.n_iter
        for key, x in self.item_d.items():
            if isinstance(x, Dict):
                keys_sample.append(key)
                self.item_d[key] = Sampler(x, total_generation_times)

        list_config = []
        for n in range(total_generation_times):
            x = {'ID': n}
            x = {**x, **{key: self.item_d[key] for key in keys_fixed}}
            for idx, key in enumerate(keys_grid):
                x[key] = product_grid[n % len(product_grid)][idx]
            for key in keys_sample:
                x[key] = self.item_d[key]()
            for key, value in x.items():
                if isinstance(value, Condition): x[key] = value(x)
            list_config.append(x)
        random.shuffle(list_config)
        return list_config

    def merge(self, params):
        """merge with already sampled parameters
        """
        config = self.make_configs()[0]; new_config = {}
        param_key = set(x.rpartition('__')[0] for x in params.param_distributions[0])
        for k,v in config.items():
            if k in param_key and isinstance(v, dict):
                for kv, vv in v.items():
                    new_config['%s_%s'%(k,kv)] = vv
            else:
                new_config[k] = v
        res = []
        for param in params:
            res.append({**new_config, **param})
        return res



