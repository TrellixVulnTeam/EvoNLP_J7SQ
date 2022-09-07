from .base import Callback
import numpy as np

__all__ = ['LossEpoch']

class LossEpoch(Callback):
    """Rewrite skorch::scoring::PassthroughScoring
    
    In the original design, a `name` is specified to group record in a batch
    I'd like to yield all record in batch
    """
    def __init__(self, on_train=True):
        self.on_train = on_train
        self.mode = 'train' if on_train else 'valid'
    
    # pylint: disable=unused-argument,arguments-differ
    def on_epoch_end(self, net, **kwargs):
        hist = net.history
        if hist[-1,'epoch'] == 1:
            try:
                self.keys = net.module_.loss_d.keys()
            except AttributeError:
                self.keys = ['loss']
        try:
            bs_key = 'train_batch_size' if self.on_train else 'valid_batch_size'
            weights, *scores = list(zip(*hist[-1, 'batches', :, [bs_key, *self.keys]]))
        except KeyError:
            return # return if there is no valid-loss getting recorded 
        score_d = {k: np.average(
            np.ma.MaskedArray(s, mask=np.isnan(s)), weights=weights
            ) for k,s in zip(self.keys, scores)}
        for name, score in score_d.items():
            hist.record(f'{self.mode}_{name}', score)

