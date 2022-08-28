from functools import wraps
from inspect import getmembers, isfunction, getfullargspec, ismodule


__all__ = ['match_args', 'doc_string']

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