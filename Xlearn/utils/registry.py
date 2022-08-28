from importlib.util import find_spec
from typing import Callable, Dict, Any, List, Optional
from functools import update_wrapper

__all__ = ['_TRANSFORMERS_AVAILABLE', '_Registry']


def _module_available(module_path: str) -> bool:
    """Check if a path is available in your environment
    >> _module_available('os') == True
    """
    try:
        return find_spec(module_path) is not None
    except ModuleNotFoundError:
        return False

_TRANSFORMERS_AVAILABLE = _module_available("transformers")

class _Registry:
    
    def __init__(self, name):
        self.name = name
        self.functions: List[Dict[str, Any]] = []
    
    def __len__(self):
        return len(self.functions)

    def __contains__(self, key):
        return any(key==e['name'] for e in self.functions)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, functions={self.functions})"

    def get(self, key, return_all=False, **metadata):
        """This function is used to gather matches from the registry:
        
        Args:
            key: Name of the registered function.
            return_all: Whether to  return all matches or just one
            metadata: Use metadata to filter against existing registered functions
        """
        matches = [e for e in self.functions if key == e["name"]]
        if not matches:
            raise KeyError(f"Key: {key} is not in {type(self).__name__}. Available keys: {self.available_keys()}")
        if metadata:
            matches = [m for m in matches if metadata.items() <= m["metadata"].items()]
            if not matches:
                raise KeyError('found no matches that fit your metadata criteria')
        return matches if return_all else matches[0]

    def remove(self, key: str):
        self.functions = [f for f in self.functions if f["name"] != key]

    def available_keys(self):
        return sorted(v["name"] for v in self.functions)
    
    def _find_matching_index(self, item: Dict[str, Any]):
        for idx, fn in enumerate(self.functions):
            if all(fn[k] == item[k] for k in ("fn", "name", "metadata")):
                return idx
        
    def _register_function(self, fn: Callable, name: Optional[str], override=False, **metadata):
        if not callable(fn): raise Exception(f'func {fn} to be registered is not callable')
        if name is None: 
            name = fn.func.__name__ if hasattr(fn, 'func') else fn.__name__    
        item = {"fn": fn, "name": name, "metadata": metadata or {}}
        idx = self._find_matching_index(item)
        if override and idx is not None:
            self.functions[idx] = item
        else:
            if idx is not None:
                raise Exception(f'func {name} is already registered, use `override=True` to redefine this function')
            self.functions.append(item)
        
    def __call__(self, fn: Optional[Callable[..., Any]]=None, name: Optional[str]=None, override=False, **metadata):
        """This function is used to register new functions to the registry 
        """
        if fn is not None:
            self._register_function(fn, name, override, **metadata)
        def _register(cls):
            self._register_function(cls, name, override, **metadata)
            return cls
        return _register



        