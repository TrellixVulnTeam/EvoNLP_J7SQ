from xxlearn.callback.base import Callback
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    LambdaLR as Lambda,
    CosineAnnealingLR,
    ExponentialLR,
    MultiStepLR,
    StepLR,
    CyclicLR
)
import torch, sys
import numpy as np
from functools import update_wrapper


__all__ = ['LRScheduler', 'LambdaLR']

class LRScheduler(Callback):
    """Callback that sets the learning rate of each
    parameter group according to some policy.

    Parameters
    ----------

    policy : str or _LRScheduler class (default='WarmRestartLR')
      Learning rate policy name or scheduler to be used.

    monitor : str or callable (default=None)
      Value of the history to monitor or function/callable. In
      the latter case, the callable receives the net instance as
      argument and is expected to return the score (float) used to
      determine the learning rate adjustment.

    event_name: str, (default='event_lr')
      Name of event to be placed in history when the scheduler takes a step.
      Pass ``None`` to disable placing events in history.
      **Note:** This feature works only for pytorch version >=1.4

    step_every: str, (default='epoch')
      Value for when to apply the learning scheduler step. Can be either 'batch'
       or 'epoch'.

    kwargs
      Additional arguments passed to the lr scheduler.

      
    Example:
    """

    def __init__(
        self, policy='WarmRestartLR', monitor='train_loss', event_name="event_lr", step_every='epoch', **kwargs
        ):
        self.policy = policy
        self.monitor = monitor
        self.event_name = event_name
        self.step_every = step_every
        vars(self).update(kwargs)

    def simulate(self, steps, initial_lr, decimals=8):
        """
        Simulates the learning rate scheduler.

        Args:
        :steps int: Number of steps to simulate
        :initial_lr float: Initial learning rate
        :decimals int: Number of rounding decimals (default=8)
         
        Returns
        :lrs ndarray: Simulated learning rates

        """
        test = torch.ones(1, requires_grad=True)
        opt = torch.optim.SGD([{'params': test, 'lr': initial_lr}])
        policy_cls = self._get_policy_cls()
        sch = policy_cls(opt, **self.kwargs)

        lrs = []
        for _ in range(steps):
            opt.step()  # suppress warning about .step call order
            lrs.append(opt.param_groups[0]['lr'])
            sch.step()

        return np.around(lrs, decimals=decimals)

    def initialize(self):
        self.policy_ = self._get_policy_cls()
        self.lr_scheduler_ = None
        self.batch_idx_ = 0
        return self

    def _get_policy_cls(self):
        if isinstance(self.policy, str):
            return getattr(sys.modules[__name__], self.policy)
        return self.policy

    @property
    def kwargs(self):
        # These are the parameters that are passed to the
        # scheduler. Parameters that don't belong there must be
        # excluded.
        excluded = ('policy', 'monitor', 'event_name', 'step_every')
        kwargs = {key: val for key, val in vars(self).items()
                  if not (key in excluded or key.endswith('_'))}
        return kwargs

    def on_train_begin(self, net, **kwargs):
        if net.history:
            try:
                self.batch_idx_ = sum(net.history[:, 'train_batch_count'])
            except KeyError:
                self.batch_idx_ = sum(len(b) for b in net.history[:, 'batches'])
        self.lr_scheduler_ = self._get_scheduler(
            net, self.policy_, **self.kwargs
        )

    def on_epoch_end(self, net, **kwargs):
        if self.step_every != 'epoch':
            return
        if isinstance(self.lr_scheduler_, ReduceLROnPlateau):
            if callable(self.monitor):
                score = self.monitor(net)
            else:
                try:
                    score = net.history[-1, self.monitor]
                except KeyError as e:
                    raise ValueError(
                        f"'{self.monitor}' was not found in history. A "
                        f"Scoring callback with name='{self.monitor}' "
                        "should be placed before the LRScheduler callback"
                    ) from e

            self.lr_scheduler_.step(score)
            # ReduceLROnPlateau does not expose the current lr so it can't be recorded
        else:
            if self.event_name is not None and hasattr(
                    self.lr_scheduler_, "get_last_lr"):
                net.history.record(self.event_name,
                                   self.lr_scheduler_.get_last_lr()[0])
            self.lr_scheduler_.step()

    def on_batch_end(self, net, training, **kwargs):
        if not training or self.step_every != 'batch':
            return
        if self.event_name is not None and hasattr(
                self.lr_scheduler_, "get_last_lr"):
            net.history.record_batch(self.event_name,
                                     self.lr_scheduler_.get_last_lr()[0])
        self.lr_scheduler_.step()
        self.batch_idx_ += 1

    def _get_scheduler(self, net, policy, **scheduler_kwargs):
        """Return scheduler, based on indicated policy, with appropriate
        parameters.
        """
        if policy not in [ReduceLROnPlateau] and \
                'last_epoch' not in scheduler_kwargs:
            last_epoch = len(net.history) - 1
            scheduler_kwargs['last_epoch'] = last_epoch

        return policy(net.optimizer_, **scheduler_kwargs)

class LambdaLR:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, func):
        lr_kwargs= self.kwargs
        lr_scheduler = LRScheduler(policy='Lambda', lr_lambda=func, **lr_kwargs)
        update_wrapper(lr_scheduler, func)
        return lr_scheduler


if __name__ == '__main__':
    from doctest import testmod
    testmod()
