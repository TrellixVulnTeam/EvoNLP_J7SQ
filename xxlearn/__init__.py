from .cli import *
from .utils import _Registry, seed_all, SEEDS, pickle_load, to_tensor
from .constant import *
from .networks import classify, regressor
from .configs import *
from .exp import run_experiment
import coloredlogs, logging, sys
coloredlogs.install(level='INFO')

__all__ = ['run_experiment', 'SEEDS', '__warehouse__', 'seed_all', 'get_logger'] + getattr(sys.modules[__name__], 'cli').__all__


def get_logger(name=None, message_only=True, formatter=None, path=''):
    logger = logging.getLogger(name or __name__)
    if message_only:
        if formatter is not None:
            formatter = logging.Formatter(formatter)
        else:
            formatter = logging.Formatter('%(asctime)s |=> %(message)s', "%H:%M")
        if path:
            log = logging.FileHandler(path,mode='w')
        else:
            log = logging.StreamHandler()
        log.setFormatter(formatter)
        logger.addHandler(log)
        # propagate=False: disable message spawned by child logger pass to parent logger
        logger.propagate = False        
    return logger