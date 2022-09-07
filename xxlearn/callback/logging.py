from typing import List
from .base import Callback
from comet_ml import Experiment, OfflineExperiment
from time import time as timer
from skorch.callbacks import ProgressBar, PrintLog, EpochScoring, PassthroughScoring
import re, numpy as np
from optuna import trial, TrialPruned
from contextlib import contextmanager
from xxlearn.utils import to_device
import pandas as pd

__all__ = [
    'CometCallback', 'EpochTimer', 'ProgressBar', 'PrintLog', 'OptunaCallback',
    'EpochScoring', 'PassthroughScoring', 'SearchCallback'
]

class CometCallback(Callback):
    def __init__(self, name, key, store='', **kwargs):
        self.name = name
        self.key = key
        self.store = store
        self.exp = None

    def initialize(self):
        if self.store == '':    #online experiment
            self.exp = Experiment(api_key=self.key, project_name=self.name)
        else:
            "see here: https://www.comet.com/docs/python-sdk/offline-experiment/"
            self.exp = OfflineExperiment(project_name=self.name, offline_directory=self.store)
        self.exp.log_text("Comet ML started")


    def on_batch_end(self, net, batch=None, training=None, **kwargs):
        """
        :batch  List[Tensor]:   input and target
        :training bool:         is it trianing or testing
        :kwargs:                loss and prediction information
        :net nn.Module instance:
            history:    [
                {'batches': [{'trian_loss': xx, 'train_batch_size': x, 'event_lr': xx}], 'epoch': 1}
            ]
            callbacks
        """
        pass

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        hist = net.history
        if (epoch := hist[-1, 'epoch']) == 1:
            self.best_key = [name for name in hist[-1].keys() if name.endswith('_best')]
            self.record_key = [k for k,v in hist[-1].items() if isinstance(v, float)]
        for key in self.best_key:
            if hist[-1, key] == True:
                self.exp.log_text(
                    "better model found at epoch %d with validation loss %5.10f"%(hist[-1, 'epoch'], hist[-1, key.rstrip('_best')])
                )
        for key in self.record_key:
            self.exp.log_metric(key, hist[-1, key], epoch=epoch)
       
    def on_train_end(self, net, **kwargs):
        #TODO: what do you want to add before end of experiment, save model/history?
        self.exp.end()


def filter_log_keys(keys, keys_ignored=None):
    """Filter out keys that are generally to be ignored.

    :keys iter(str):  all keys
    :keys_ignored iter(str): keys to be ignored, default=None
    """
    keys_ignored = keys_ignored or ()
    for key in keys:
        if not (
                key == 'epoch' or
                (key in keys_ignored) or
                key.endswith('_best') or
                key.endswith('_batch_count') or
                key.startswith('event_')
        ):
            yield key


class EpochTimer(Callback):
    """Measures the duration of each epoch and writes it to the
    history with the name ``dur``.
    """
    def __init__(self, **kwargs):
        super(EpochTimer, self).__init__(**kwargs)

        self.epoch_start_time_ = None

    def on_epoch_begin(self, net, **kwargs):
        self.epoch_start_time_ = timer()

    def on_epoch_end(self, net, **kwargs):
        net.history.record('dur', timer() - self.epoch_start_time_)


class OptunaCallback(Callback):
    """Skorch callback to prune unpromising trials.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g. ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries,
            i.e., ``net.histroy``. The names thus depend on how this dictionary
            is formatted.
    """
    class PseudoTrial:
        def report(self,metric, epoch): pass
        def should_prune(self): return False
        def set_user_attr(self,k, lst): pass

    def __init__(self, trial: trial.Trial, monitor: str) -> None:
        super().__init__()
        self._trial = trial or self.PseudoTrial()
        self._monitor = monitor

    def on_epoch_end(self, net, **kwargs):
        history = net.history
        if not history:
            return
        epoch = len(history) - 1
        current_score = history[-1, self._monitor]
        self._trial.report(current_score, epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise TrialPruned(message)

    @staticmethod
    def obtain_key(keys: List[str]):
        pat = re.compile(r"^(train|test|valid)_[A-Za-z0-9]+$")
        return filter(pat.match, keys)

    def on_train_end(self, net, X=None, y=None, **kwargs):
        history = net.history
        for key in self.obtain_key(history[-1].keys()):
            self._trial.set_user_attr(key, [dic[key] for dic in history])

class Score:
    def __init__(self, name, lower_is_better, _scoring):
        self.name = name
        self.lower_is_better = lower_is_better
        self._scoring = _scoring
        self.best_score_ = np.inf if lower_is_better else -np.inf

    def _is_best_score(self, current_score):
        if self.lower_is_better:
            if current_score < self.best_score_:
                self.best_score_ = current_score
        else:
            if current_score > self.best_score_:
                self.best_score_ = current_score
        return current_score == self.best_score_
    
    def __call__(self, net, X, y=None): 
        return self._scoring(net, X, y)

class SearchCallback(Callback):

    def __init__(self, valid_data=None, test_data=None):
        self.valid_data, self.test_data = valid_data, test_data
    
    def on_train_begin(self, net, X=None, y=None, **kwargs):
        self.valid_scores, self.test_scores = {}, {}
        def initialize_score(mode, name, cb):
            if mode == 'valid':
                self.valid_scores[name] = Score(f'valid_{name}', cb.lower_is_better, cb._scoring)
            else:
                self.test_scores[name] = Score(f'test_{name}', cb.lower_is_better, cb._scoring)
        for name, cb in net.callbacks_:
            if isinstance(cb, EpochScoring):
                name = cb.name.split('_')[-1]
                self.valid_data is not None and initialize_score('valid', name, cb)
                self.test_data is not None and initialize_score('test', name, cb)
    
    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        hist = net.history
        for name, score, better in get_scores(
            self.valid_data, self.valid_data.y, net, self.valid_scores):
            hist.record(f'valid_{name}', score)
            hist.record(f'valid_{name}_best', better)
        for name, score, better in get_scores(
            self.test_data, self.test_data.y, net, self.test_scores):
            hist.record(f'test_{name}', score)
            hist.record(f'test_{name}_best', better)



@contextmanager
def _cache_net_predict_proba(net, y_pred):
    "make `predict_proba` to always produce iter(y_pred)"
    # pylint: disable=unused-argument
    def cached_predict_proba(*args, **kwargs):
        return y_pred
    net.predict_proba = cached_predict_proba
    try:
        yield net
    finally:
        #undo overriding predict_proba method
        del net.__dict__['predict_proba']

def get_scores(X, y_true, net, score_d):
    "a generator"
    if X is None: return ()
    y_pred = net.predict_proba(X)
    with _cache_net_predict_proba(net, y_pred) as cache_net:
        for name, score_f in score_d.items():
                score = score_f(cache_net, X, y_true)
                is_best = score_f._is_best_score(score)
                yield name, score, is_best