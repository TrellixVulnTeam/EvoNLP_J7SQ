from skorch.helper import SliceDataset
from Xlearn.utils import to_numpy
from collections.abc import Mapping
import numpy as np

class Xdata(SliceDataset):

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