from skorch import NeuralNetClassifier, NeuralNetRegressor
from types import FunctionType
from functools import update_wrapper, wraps

__all__ = ['classify', 'regressor']

class base:
    """Skorch Wrapper (see https://skorch.readthedocs.io/en/latest/classifier.html for detail)

    Args:                                               \n
    :module     :       Torch nn module                 \n
    :criterion  :       default=nn.NLLLoss              \n
    :clsses     :       default=None, infer from `y` passed to fit()    \n
    :optimizer  :       default=optim.SGD               \n
    :lr         :       default=1e-2                    \n
    :max_epochs :       default=10                      \n
    :batch_size :       default=128, entire data is fit if set to be -1 \n
    :iterator_train | iterator_valid | dataset :        \n
    :train_split:       default=skorch.dataset.ValidSplit(5)    \n
    :callbacks  :       see get_default_callbacks(); disable all callbacks if set to be `disable`   \n
    :warm_start :       whether each fit() leads to re-initialization of module (cold start) or whether oduleshould be trained further (warm start) \n
    :device     :       which device used to compute    \n
    """

    def __init__(self, module, **kwargs):
        self.kwargs = kwargs
        self.base = module

    def __call__(self, cls):
        nn_kwargs = self.kwargs
        @wraps(cls, updated=())
        class wrapped(self.base):
            def __init__(self, *args, **kwargs):
                inst = cls(*args, **kwargs)
                super(wrapped, self).__init__(module=inst, **nn_kwargs)
            def silence(self):
                self.verbose = 0
        # self.update_doc(cls, wrapped)
        wrapped.__wrapped__ = None
        return wrapped

    def update_doc(self, from_cls, to_cls):
        update_wrapper(to_cls, from_cls)
        for name, func in vars(from_cls).items():
            if isinstance(func, FunctionType) and func.__doc__ is not None:
                update_wrapper(getattr(to_cls, name), func)

class classify(base):
    def __init__(self, **kwargs):
        super(classify, self).__init__(NeuralNetClassifier, **kwargs)

class regressor(base):
    def __init__(self, **kwargs):
        super(regressor, self).__init__(NeuralNetRegressor, **kwargs)




    
if __name__ == "__main__":
    from doctest import testmod
    testmod()