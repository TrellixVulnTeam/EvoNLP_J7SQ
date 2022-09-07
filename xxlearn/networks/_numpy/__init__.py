import numpy as np
 
__all__ = ['xzip', 'xpad']


def xzip(*args, fillvalue=None):
    """
    >>> a, b = 'ABCD', 'xy'
    >>> xzip(a, b, fillvalue='-')
    Ax By C- D-
    """
    iterators = [iter(it) for it in args]
    num_active = len(iterators)
    if not num_active:
        return
    while True:
        values = []
        for i, it in enumerate(iterators):
            try:
                value = next(it)
            except StopIteration:
                num_active -= 1
                if not num_active: return
                iterators[i] = repeat(fillvalue)
                value = fillvalue
            values.append(value)
        yield tuple(values)

def xpad(arr, *shape, pad_value=0, dtype=float, ):
    if not shape: return 
    _arr = np.full(shape, fill_value=pad_value, dtype=dtype)
    for i, x in enumerate(arr):
        if isinstance(x, np.ndarray):
            size = min(shape[1], len(x))
            _arr[i, :size] = x[:size]
        else:
            _arr[i, :shape[1]] = xpad(x, *shape[1:], pad_value=pad_value)
    return _arr


def repeat(obj, times=None):
    """
    >>> repeat(10, 3)
    10 10 10
    """
    if times is None:
        while True: yield obj
    else:
        for _ in range(times): yield obj