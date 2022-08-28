from pathlib import Path
import os

__all__ = ['__warehouse__']
__warehouse__ = Path(os.path.expanduser("~")).joinpath("scikit_learn_data")
__datasets__ = ['Lastfm1k', 'FourSquare-NYC', 'FourSquare-Tokyo', 'Tianchi']