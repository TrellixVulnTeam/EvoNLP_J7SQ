from functools import reduce
from typing import Iterable
import numpy as np
import torch

__all__ = []

def tween(lst, item, add_last=False):
    """
    >>> a, b = [1,2,3], [#,$]
    >>> tween(a,b) == [1,#,$,2,#,$,3]
    >>> tween(a,b,True) == [1,#,$,2,#,$,3,#,$]
    """
    if not isinstance(item, list):
        item = [item]
    if add_last:
        return reduce(lambda r,v: r+[v]+item, lst, [])
    else:
        return reduce(lambda r,v: r+item+[v], lst[1:], lst[:1])

def _flatten(lst):
    for x in lst:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from _flatten(x)
        else:
            yield x

def flatten(lst, rtn_lst=True):
    if rtn_lst:
        return list(_flatten(lst))
    else:
        return _flatten(lst)


def seq_len_to_mask(seq_len, max_len=None):
    """
    >>> size = torch.randint(3, 10, (3,)) # [3,6,6]
    >>> seq_len_to_mask(size)             # shape = (3,6) True/False matrix
    >>> seq_len_to_mask(size, 10)         # shape = (3,10) True/False matrix
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask