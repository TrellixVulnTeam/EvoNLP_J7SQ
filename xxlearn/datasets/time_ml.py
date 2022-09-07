from xxlearn.datasets.base import Dataset
from xxlearn.utils import ofind, cache_results
from xxlearn import __warehouse__
from torchvision.datasets.mnist import read_image_file, read_label_file
import numpy as np

__all__ = ['TIME']

class TIME(Dataset):

    __name__ = 'TIME'

    @cache_results(_cache_fp=__warehouse__.joinpath('TIME', 'data.pkl'))
    def load_data(self, _refresh=False, category=None):
        data = []
        for data_path in ofind(self.process_folder, '-images-'):
            data.append(read_image_file(data_path).numpy().astype('float32'))
        data = np.concatenate(data, axis=0)
        target = []
        for target_path in ofind(self.process_folder, '-labels-'):
            target.append(read_label_file(target_path).numpy().astype('int64'))
        target = np.concatenate(target, axis=0)
        return data, target

