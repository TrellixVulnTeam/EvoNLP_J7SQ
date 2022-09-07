from sklearn.model_selection._search import ParameterSampler
from typing import Iterable, Mapping
from .other import _is_distribution, Grid, Dist
from copy import deepcopy

__all__ = ['Sampler']

class Sampler(ParameterSampler):
    
    def __init__(self, param_distributions, n_iter, *, random_state=None):
        self.sample_keys = []
        self._param_distributions = deepcopy(param_distributions)
        for k, v in param_distributions.items():
            if _is_distribution(v): 
                self.sample_keys.append(k); continue
            elif isinstance(v, Grid):
                self.sample_keys.append(k); param_distributions[k] = v * n_iter
            else:
                param_distributions[k] = Dist(v, name=k)
        super(Sampler, self).__init__(param_distributions, n_iter)
        self.iter = iter(self)
        self.generated = []

    def __call__(self):
        rtn = next(self.iter)
        self.generated.append(rtn)
        return rtn

    @property
    def key(self):
        return self.sample_keys

    # def __getitem__(self, __name): return self._param_distributions[__name]

    def retrieve_from_lst(self):
        for key in self.key:
            dist = self._param_distributions[key]
            if _is_distribution(dist):
                lst = [x[key] for x in self.generated]
                yield key, [min(lst), max(lst)]
            else:
                yield key, dist

