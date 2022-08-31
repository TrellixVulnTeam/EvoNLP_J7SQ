from Xlearn.datasets.base import Dataset
from Xlearn.utils import ofind, cache_results 
from Xlearn import __warehouse__, get_logger
import numpy as np
from dataclasses import dataclass

__all__ = ['PTB']
__warehouse__ = __warehouse__.joinpath('ML')
logger = get_logger()

@dataclass
class ML(Dataset):
    __name__ = 'ML'

    def __init__(self, rootdir=None):
        self.__name__ += '/' + self.name
        super(ML, self).__init__(rootdir)

    def __post_init__(self):
        logger.info('load data from ', self.rootdir)

    def _donwload_and_extract(self):
        pass

    def load_data(self, _refresh=False):
        return cache_results(_cache_fp=__warehouse__.joinpath(self.name, 'data.pkl'), _refresh=_refresh)(self._load_data)()



class PTB(ML):
    name: str = 'PTB'
    __url__ = 'https://data.deepai.org/ptbdataset.zip'

    __corefile__ = {
        "train": "train.txt",
        "valid": "valid.txt",
        "test": "test.txt",
    }

    def _load_data(self):
        from dstoolbox.transformers import TextFeaturizer
        t = TextFeaturizer(lowercase=True, stop_words='english')
        data = []
        for cat, name in self.__corefile__.items():
            with open(self.rootdir.joinpath(name), 'r') as f:
                data.extend([line + ' <eos>' for line in f])
        res = t.fit_transform(data)
        return res, t.vocabulary_

            

