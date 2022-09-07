from abc import ABCMeta, abstractmethod
from enum import Enum

__all__ = ['Number', 'Color']

class Number(metaclass=ABCMeta):
    __slots__ = ()
    __hash__ = None
    
class Color(Enum):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'
