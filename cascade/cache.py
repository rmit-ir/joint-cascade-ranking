import logging
from pathlib import Path

import sklearn.datasets

from sklearn.externals import joblib

location = Path(__file__).resolve().parent.parent / '.cache'
location = str(location)
mem = joblib.Memory(location=location, verbose=logging.DEBUG)


@mem.cache
def load_svmlight_file(*args, **kwargs):
    return sklearn.datasets.load_svmlight_file(*args, **kwargs)
