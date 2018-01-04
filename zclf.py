"""
zlib classifier module
"""

import zlib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm
from typing import Callable


class ZlibClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier based on zlib
    """

    def __init__(self, encoder: Callable[..., bytes]):
        self.encoder = encoder

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self._labels = unique_labels(y)

        # Prepare per class bytes
        self._bytesX = { c: [] for c in self._labels }
        self._clensX = { c: [] for c in self._labels }
        for x_inst, y_inst in tqdm(zip(X, y)):
            self._bytesX[y_inst].append(self.encoder(x_inst))

        # Merge all instances and calculate compressed lengths
        for l in self._labels:
            self._bytesX[l] = b"".join(self._bytesX[l])
            self._clensX[l] = len(zlib.compress(self._bytesX[l]))

        return self

    def _zlib_dist(self, label, test: bytes) -> float:
        compressed = zlib.compress(self._bytesX[label] + test)
        return (len(compressed) - self._clensX[label]) / len(test)

    def predict(self, X):
        check_is_fitted(self, ["_bytesX", "_clensX", "_labels"])

        labels = []
        for x in tqdm(X):
            encoded = self.encoder(x)
            labels.append(min(self._labels, key=lambda l: self._zlib_dist(l, encoded)))
        return labels
