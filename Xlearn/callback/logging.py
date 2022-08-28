from typing import List
from .base import Callback
from comet_ml import Experiment, OfflineExperiment
from time import time as timer
from tqdm import tqdm, tqdm_notebook
import numpy as np
from tabulate import tabulate
from Xlearn.utils.types import Number, Color
from skorch.callbacks import ProgressBar, PrintLog, EpochScoring, PassthroughScoring
from itertools import cycle
import sys, re
from optuna import trial, TrialPruned

__all__ = [
    'CometCallback', 'EpochTimer', 'ProgressBar', 'PrintLog', 'OptunaCallback',
    'EpochScoring', 'PassthroughScoring'
]

class CometCallback(Callback):
    def __init__(self, name, key, store='', **kwargs):
        self.name = name
        self.key = key
        self.store = store
        self.experiment = None

    def initialize(self):
        if self.store == '':    #online experiment
            self.experiment = Experiment(api_key=self.key, project_name=self.name)
        else:
            "see here: https://www.comet.com/docs/python-sdk/offline-experiment/"
            self.experiment = OfflineExperiment(project_name=self.name, offline_directory=self.store)

    def on_train_begin(self, net, **kwargs):
        pass

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

    def on_train_end(self, net, **kwargs):
        train_loss = net.history[-1, "train_loss"]
        valid_loss = net.history[-1, "valid_loss"]
        self.experiment.log_metrics(
            {"train_loss": train_loss, "valid_loss": valid_loss}
        )
        self.experiment.end()


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