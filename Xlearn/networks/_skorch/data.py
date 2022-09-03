from skorch.helper import SliceDataset
from torch.utils.data import ConcatDataset
from sklearn.model_selection import PredefinedSplit
from Xlearn.utils import to_numpy
from collections.abc import Mapping
import numpy as np

__all__ = ['Xdata', 'DisabledCV']

class Xdata(SliceDataset):

    def __init__(self, dataset, idx=0, indices=None):
        self._cv = 3
        if isinstance(dataset, list) and len(dataset) == 2:
            dataset = ConcatDataset(dataset)
            self._cv = PredefinedSplit(make_split(dataset.cummulative_sizes))
        super().__init__(dataset, idx, indices)

    @property
    def cv(self):
        return self._cv

    @property
    def train_split(self):
        def _make_split(X, y=None, valid_ds=None, **kwargs):
            return self.dataset.datasets
        return _make_split

    def _select_item(self, Xn):
        if isinstance(Xn, Mapping):
            return Xn
        else:
            return super(Xdata, self)._select_item(Xn)

    def __array__(self, dtype=None):
        # This method is invoked when calling np.asarray(X)
        # https://numpy.org/devdocs/user/basics.dispatch.html
        X = [self[i] for i in range(len(self))]
        if np.isscalar(X[0]):
            return np.asarray(X)
        return np.asarray([to_numpy(x) for x in X], dtype=dtype)

    def transform(self, data):
        "apply transformation to each record in dataset"
        return data

class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield (np.arange(len(X)), np.arange(len(y)))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

def make_split(sizes: list):
    """
    >>> sizes = [550, 733]
    >>> make_split(sizes)
    """
    assert len(sizes) == 2, "you must specify two value indicating ending index of train/valid split"
    idx = np.zeros(sizes[1], )
    idx[:sizes[0]] = -1
    return idx