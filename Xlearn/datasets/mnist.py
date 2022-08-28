from Xlearn.datasets.base import Dataset
from Xlearn.utils import ofind
from torchvision.datasets.mnist import read_image_file, read_label_file
import numpy as np

__all__ = ['MNIST']

class MNIST(Dataset):

    __url__ = 'http://yann.lecun.com/exdb/mnist/'
    __corefile__ = {
        "trainX": "train-images-idx3-ubyte.gz",
        "trianY": "train-labels-idx1-ubyte.gz",
        "testX": "t10k-images-idx3-ubyte.gz",
        "testY": "t10k-labels-idx1-ubyte.gz"
    }
    __name__ = 'MNIST'

    def load_data(self):
        data = []
        for data_path in ofind(self.process_folder, '-images-'):
            data.append(read_image_file(data_path).numpy().astype('float32'))
        data = np.concatenate(data, axis=0)
        target = []
        for target_path in ofind(self.process_folder, '-labels-'):
            target.append(read_label_file(target_path).numpy().astype('int64'))
        target = np.concatenate(target, axis=0)
        return data, target

        
