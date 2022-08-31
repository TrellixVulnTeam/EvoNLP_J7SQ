import torch
import torch.nn as nn
from torch.nn.utils import rnn
from functools import wraps

__all__ = ['xpack']

class xpack:

    def __init__(self, data, length, batch_first=True, enforce_sorted=False):
        self.length = length
        self.data, self.batch_sizes, self.sorted_indices, self.unsorted_indices = rnn.pack_padded_sequence(data, length, batch_first=batch_first, enforce_sorted=enforce_sorted)
    
    @property
    def sequence(self):
        return rnn.PackedSequence(
            data=self.data, batch_sizes=self.batch_sizes,
            sorted_indices=self.sorted_indices, unsorted_indices=self.unsorted_indices
        )
    
    @wraps(rnn.pack_padded_sequence)
    def pad(self, data=None, batch_first=True, padding_value=0, total_length=None, **kwargs):
        if data is None: data = self.data
        batch_sizes = kwargs.get('batch_sizes', self.batch_sizes)
        sorted_indices = kwargs.get('sorted_indices', self.sorted_indices)
        unsorted_indices = kwargs.get('unsorted_indices', self.unsorted_indices)
        rtn, _ = rnn.pad_packed_sequence(
            rnn.PackedSequence(
                data, batch_sizes, sorted_indices, unsorted_indices
            ), 
            batch_first=batch_first, padding_value=padding_value, total_length=total_length
        )
        return rtn

    def pack(self, data, batch_first=True, enforce_sorted=False, return_sequence=False):
        seq =  rnn.pack_padded_sequence(data, self.length, batch_first=batch_first, enforce_sorted=enforce_sorted)
        return seq if return_sequence else seq.data

    def repeat(self, data):
        idx = torch.cat([self.sorted_indices[:i] for i in self.batch_sizes])
        return data[idx]