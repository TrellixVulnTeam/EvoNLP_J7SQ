

import os, logging, urllib
from pathlib import Path
from tqdm import tqdm
from Xlearn.constant import __warehouse__
from torchvision.datasets.utils import extract_archive

logger = logging.getLogger(__name__)

class Dataset:
    """ Base dataset of SR datasets
    """
    __name__ = ''
    __corefile__ = {'all': ''}

    def __init__(self, rootdir=None):
        """ `rootdir` is the directory of the raw dataset """
        if rootdir is not None and isinstance(rootdir, str):
            self.rootdir = Path(rootdir)
        else:
            self.rootdir = __warehouse__.joinpath(self.__name__)
        self._make_dir()
        self._donwload_and_extract()
        self.load_data()

    def load_data(self, *args, **kwargs):
        """
        Return:
            data and target
        """
        raise NotImplementedError

    @property
    def raw_folder(self):
        return self.rootdir.joinpath('raw')

    @property
    def process_folder(self):
        return self.rootdir.joinpath('processed')

    def _make_dir(self):
        self.raw_folder.mkdir(parents=True, exist_ok=True)
        self.process_folder.mkdir(parents=True, exist_ok=True)

    def _donwload_and_extract(self):
        for cat, name in self.__corefile__.items():
            this_url = self.__url__ + name
            out_path = self.raw_folder.joinpath(name)
            if out_path.exists():
                continue
            try:
                _urlretrieve(this_url, out_path)
                logger.info("Download successful")
            except:
                logger.exception("Download failed, please try again")
            extract_archive(
                from_path=out_path, to_path=self.process_folder, remove_finished=False
            )
        
    def transform(self):
        """ Transform to the general data format
        """
        raise NotImplementedError



def _save_response_content(content, destination, length=None) -> None:
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            if not chunk:   # filter out keep-alive new chunks
                continue
            fh.write(chunk)
            pbar.update(len(chunk))


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024 * 32) -> None:
    "see torchvision::datasets::utils.py"
    with urllib.request.urlopen(urllib.request.Request(url)) as response:
        _save_response_content(
            iter(lambda: response.read(chunk_size), b""), filename, length=response.length
        )
