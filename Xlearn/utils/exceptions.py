"""Contains exceptions and warnings."""

from sklearn.exceptions import NotFittedError


class Exception(BaseException):
    """Base exception."""


class NotInitializedError(Exception, NotFittedError):
    """Module is not initialized, please call the ``.initialize``
    method or train the model by calling ``.fit(...)``.

    """


class AttributeError(Exception):
    """An attribute was set incorrectly on the model."""


class Warning(UserWarning):
    """Base warning."""


class DeviceWarning(Warning):
    """A problem with a device (e.g. CUDA) was detected."""


class TrainingImpossibleError(Exception):
    """The net cannot be used for training"""
