from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.scoring import loss_scoring
from types import FunctionType
from functools import update_wrapper, wraps
from Xlearn.utils import doc_string, to_tensor, _is_slicedataset
from Xlearn.callback import EpochScoring, PassthroughScoring, EpochTimer, PrintLog, LossEpoch
from .data import *


__all__ = ['classify', 'regressor', 'Classifier', 'Xdata', 'DisabledCV']

class base:
    """Skorch Wrapper (see https://skorch.readthedocs.io/en/latest/classifier.html for detail)

    Args:                                               \n
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

    def __init__(self, module, use_amp=False, **kwargs):
        if use_amp:
            module, kwargs = get_accelerator(module, kwargs)
        self.kwargs = kwargs
        self.base = module

    def __call__(self, cls):
        nn_kwargs = self.kwargs; parent = self.base
        @doc_string(parent=cls)
        @wraps(cls, updated=())
        class wrapped(self.base):
            def __init__(self, *args, **kwargs):
                if not args and not kwargs:
                    super(wrapped, self).__init__(module=cls, **nn_kwargs)
                else:
                    super(wrapped, self).__init__(module=cls(*args, **kwargs), **nn_kwargs)
            def silence(self):
                self.verbose = 0
            @property
            def __class__(self):
                return parent
        # self.update_doc(cls, wrapped)
        wrapped.__wrapped__ = None
        return wrapped

    def update_doc(self, from_cls, to_cls):
        update_wrapper(to_cls, from_cls)
        for name, func in vars(from_cls).items():
            if isinstance(func, FunctionType) and func.__doc__ is not None:
                update_wrapper(getattr(to_cls, name), func)

def get_accelerator(module, kwargs):
    from accelerate import Accelerator
    from skorch.hf import AccelerateMixin
    accelerator = Accelerator(mixed_precision='fp16')
    kwargs['accelerator'] = accelerator
    kwargs['device'] = None
    class new_module(AccelerateMixin, module):
        """NeuralNetClassifier with accelerate support"""
    return new_module, kwargs

class Classifier(NeuralNetClassifier):

    def score(self, X, y=None):
        return loss_scoring(self, X, y)

    def use_amp(self):
        kwargs = self.get_params(deep=False)
        cls, kwargs = get_accelerator(NeuralNetClassifier, kwargs)
        new_self = cls(**kwargs)
        self.__dict__.update(new_self.__dict__)

    def get_params_for(self, prefix):
        kwargs = super().get_params_for(prefix)
        if prefix == 'iterator_train':
            kwargs = {**kwargs, 'shuffle': True}
        return kwargs

    def set_default_callback(self, *names):
        """Disable all default callbacks, this should be called before fit
        """
        callbacks = []
        for name, cb, kwargs in self.__get_default_callbacks():
            if name in names:
                callbacks.append((name, cb(**kwargs)))
        def get_default_callbacks(): return callbacks
        self.get_default_callbacks = get_default_callbacks
        
    def __get_default_callbacks(self):
        yield from [
            ('valid_acc', EpochScoring, dict(scoring='accuracy', name='valid_acc', lower_is_better=False)),
            ('epoch_timer', EpochTimer, dict()),
            ('print_log', PrintLog, dict()),
            ('f1_binary', EpochScoring, dict(scoring='f1', lower_is_better=False, name='valid_f1', on_train=False, use_caching=True)),
            ('prec_binary', EpochScoring, dict(scoring='precision', lower_is_better=False, name='valid_prec', on_train=False, use_caching=True)),
            ('rec_binary', EpochScoring, dict(scoring='recall', lower_is_better=False, name='valid_rec', on_train=False, use_caching=True)),
            ('train_loss', LossEpoch, dict(on_train=True)),
            ('valid_loss', LossEpoch, dict(on_train=False)),
        ]
    
    def infer(self, x, **fit_params):
        """Overwrite skorch::NeuralNet::infer to allow flexible type of inputs
        """
        x = to_tensor(x, device=self.device)
        if isinstance(x, dict):
            x_dict = self._merge_x_and_fit_params(x, fit_params)
            return self.module_(**x_dict)
        return self.module_(x, **fit_params)

    def get_dataset(self, X, y=None):
        if _is_slicedataset(X):
            return X
        return super().get_dataset(X, y)

    def check_training_readiness(self):
        self.__pre_init__()
        return super().check_training_readiness()
    
    def __pre_init__(self):
        pass

    def fit(self, X, y, **fit_params):
        return super().fit(X, y, **fit_params)

class classify(base):
    def __init__(self, **kwargs):
        super(classify, self).__init__(NeuralNetClassifier, **kwargs)
        self.__name__ = 'Classifier'

class regressor(base):
    def __init__(self, **kwargs):
        super(regressor, self).__init__(NeuralNetRegressor, **kwargs)
        self.__name__ = 'Regressor'

    
if __name__ == "__main__":
    from doctest import testmod
    testmod()