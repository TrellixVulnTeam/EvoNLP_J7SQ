from functools import wraps
from inspect import getmembers, isfunction, getfullargspec, ismodule, signature as s_
import logging, os
from Xlearn.utils.io import pickle_dump, pickle_load

logger = logging.getLogger(__file__)

__all__ = ['match_args', 'doc_string', 'cache_results']

def params_for(d, prefix=''):
    """
    >>> kwargs = {'encoder__a': 3, 'encoder__b': 4, 'decoder__a': 5}
    >>> params_for(kwargs, 'encoder')
    {'a': 3, 'b': 4}

    """
    if prefix == '': return d
    if not prefix.endswith('__'): prefix += '__'
    return {
        key[len(prefix):]: val for key, val in d.items() if key.startswith(prefix)
    }
    
def match_args(func, prefix='', **kwargs):
    """
    >>> class A:
    ...     def __init__(self, name):
    ...         self.name = name
    >>> match_args(A, 'module', **{'module__name': 'bert'})
    {'name': 'bert'}
    """
    spect = getfullargspec(func)
    if spect.varkw is not None: 
        return kwargs
    needed_args = set(spect.args)
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    else:
        defaults = []
    start_idx = len(spect.args) - len(defaults)
    output = {name: default for name, default in zip(spect.args[start_idx:], defaults)}
    output.update({
        name: val for name, val in params_for(kwargs, prefix).items() if name in needed_args
    })
    return output


def cache_results(_cache_fp, _refresh=False):
    r""" a decorator to cache data-loader
    @reference: FastNLP::core::utils
    
    :param str `_cache_fp`:     where to read the cache from
    :param bool `_refresh`:     whether to regenerate cache

    >>> @cache_results('/tmp/cache.pkl')
    ... def load_data():
        # some time-comsuming process
        return processed_data
    """

    def wrapper_(func):
        signature = s_(func)

        def wrapper(*args, **kwargs):
            cache_filepath = kwargs.pop('_cache_fp', _cache_fp)
            refresh = kwargs.pop('_refresh', _refresh)
            refresh_flag = True

            if cache_filepath is not None and refresh is False:
                if os.path.exists(cache_filepath):
                    results = pickle_load(cache_filepath)
                    refresh_flag = False

            if refresh_flag:
                results = func(*args, **kwargs)
                if cache_filepath is not None:
                    if results is None:
                        raise RuntimeError("The return value is None. Delete the decorator.")
                    _prepare_cache_filepath(cache_filepath)
                    pickle_dump(results, cache_filepath)
                    logger.info("Save cache to {}.".format(cache_filepath))

            return results

        return wrapper

    return wrapper_

def _prepare_cache_filepath(filepath):
    _cache_filepath = os.path.abspath(filepath)
    if os.path.isdir(_cache_filepath):
        raise RuntimeError("The cache_file_path must be a file, not a directory.")
    cache_dir = os.path.dirname(_cache_filepath)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

def doc_string(parent=None):
    flag = (parent is None)
    def _doc_string(cls):
        p_cls = parent
        for name, func in getmembers(cls, isfunction):
            if func.__doc__: continue
            for p in cls.__mro__[1:]:
                if flag: 
                    p_cls = p
                if hasattr(p_cls, name):
                    func.__doc__ = getattr(p_cls, name).__doc__
        if p_cls and not cls.__doc__:
            cls.__doc__ = p_cls.__doc__
        return cls
    return _doc_string

if __name__ == '__main__':
    from doctest import testmod
    testmod()