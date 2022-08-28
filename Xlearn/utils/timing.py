import functools
from contextlib import contextmanager

from time import perf_counter
from datetime import timedelta
from datetime import datetime
from colorama import Fore, Style


__all__ = ['timeit', 'color_str']

@contextmanager
def timed(color='green', bold=False):
    r"""A decorator to print the total time of executing a body function. 
    
    Args:
        color (str, optional): color name. Default: 'green'
        bold (bool, optional): if ``True``, then the verbose is bolded. Default: ``False``
    """
    t = perf_counter()
    yield
    total_time = timedelta(seconds=round(perf_counter() - t))
    timestamp = datetime.now().isoformat(' ', 'seconds')
    print(color_str(string=f'\nTotal time: {total_time} at {timestamp}', 
                    color=color, 
                    bold=bold))
    
    
def timeit(_func=None, *, color='green', bold=False):
    def decorator_timeit(f):
        r"""Print the runtime of the decorated function. """
        @functools.wraps(f)
        def wrapper_timeit(*args, **kwargs):
            t = perf_counter()
            out = f(*args, **kwargs)
            total_time = timedelta(seconds=round(perf_counter() - t))
            timestamp = datetime.now().isoformat(' ', 'seconds')
            print(color_str(string=f'\nTotal time: {total_time} at {timestamp}', 
                            color=color, 
                            bold=bold))
            return out
        return wrapper_timeit
    if _func is None:
        return decorator_timeit
    else:
        return decorator_timeit(_func)


def color_str(string, color, bold=False):
    r"""Returns stylized string with coloring and bolding for printing.
    
    Example::
    
        >>> print(color_str('lagom', 'green', bold=True))
        
    Args:
        string (str): input string
        color (str): color name
        bold (bool, optional): if ``True``, then the string is bolded. Default: ``False``
    
    Returns:
        out: stylized string
    
    """
    colors = {'red': Fore.RED, 'green': Fore.GREEN, 'blue': Fore.BLUE, 'cyan': Fore.CYAN, 
              'magenta': Fore.MAGENTA, 'black': Fore.BLACK, 'white': Fore.WHITE}
    style = colors[color]
    if bold:
        style += Style.BRIGHT
    out = style + string + Style.RESET_ALL
    return out