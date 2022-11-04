from xxlearn.datasets.base import Dataset
from xxlearn.utils import ofind, cache_results
from xxlearn import __warehouse__
import numpy as np
from sklearn.datasets import load_files
from fastNLP import cache_results as f

__all__ = ['IMDB']

class IMDB(Dataset):

    __url__ = 'http://ai.stanford.edu/~amaas/data/sentiment/'
    __corefile__ = {
        'all': 'aclImdb_v1.tar.gz '
    }
    __name__ = 'IMDB'

    def extract(self, from_path, to_path):
        import tarfile
        with tarfile.open(from_path, 'r:gz') as f:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, path=to_path)

    @cache_results(_cache_fp=__warehouse__.joinpath('IMDB', 'data.pkl'))
    def load_data(self, _refresh=False):
        prefix = 'aclImdb'
        train = load_files(self.process_folder.joinpath(prefix,'train'), categories=['neg', 'pos'])
        test = load_files(self.process_folder.joinpath(prefix,'test'), categories=['neg', 'pos'])
        data = train['data'] + test['data']
        target = np.concatenate([train['target'], test['target']], axis=0)
        return np.asarray([x.decode() for x in data]), target




