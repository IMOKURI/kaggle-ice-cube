# https://github.com/analokmaus/kuma_utils/blob/master/preprocessing/transformer.py

import logging
from copy import copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, PowerTransformer, QuantileTransformer, StandardScaler
from tqdm import tqdm

from .base import BaseTransformer

log = logging.getLogger(__name__)


class _DistTransformer(BaseTransformer):

    TRANSFORMS = {"standard", "min-max", "box-cox", "yeo-johnson", "rankgauss", "uniform", "ordinal"}

    def __init__(self, transform="standard"):
        assert transform in self.TRANSFORMS
        self.t = transform

    def fit(self, X: pd.Series, y=None) -> None:
        if self.t == "standard":
            self.transformer = StandardScaler()
        elif self.t == "min-max":
            self.transformer = MinMaxScaler()
        elif self.t == "box-cox":
            self.transformer = PowerTransformer(method="box-cox")
        elif self.t == "yeo-johnson":
            self.transformer = PowerTransformer(method="yeo-johnson")
        elif self.t == "rankgauss":
            self.transformer = QuantileTransformer(random_state=440, output_distribution="normal")
        elif self.t == "uniform":
            self.transformer = QuantileTransformer(random_state=440)
        elif self.t == "ordinal":
            self.transformer = OrdinalEncoder()
        else:
            raise ValueError(self.transform)

        if isinstance(X, pd.Series):
            self.transformer.fit(X.values.reshape(-1, 1))
        elif isinstance(X, np.ndarray):
            self.transformer.fit(X.reshape(-1, 1))
        else:
            raise TypeError(type(X))

    def transform(self, X: pd.Series) -> np.ndarray:
        if isinstance(X, pd.Series):
            return self.transformer.transform(X.values.reshape(-1, 1))
        elif isinstance(X, np.ndarray):
            return self.transformer.transform(X.reshape(-1, 1))
        else:
            raise TypeError(type(X))

    def fit_transform(self, X: pd.Series) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def copy(self):
        return copy(self)


class DistTransformer(BaseTransformer):
    """
    Distribution Transformer for numerical features

    Availbale transforms:
        TRANSFORMS = {
            'standard', 'min-max', 'box-cox', 'yeo-johnson', 'rankgauss', 'uniform', 'ordinal'
        }
    """

    def __init__(self, transform="standard", verbose=True):
        self.t = transform
        self.transformers = {}
        self.verbose = verbose

    def fit(self, X: pd.DataFrame) -> None:
        self._input_cols = X.columns.tolist()

        col_iter = tqdm(self._input_cols) if self.verbose else self._input_cols
        for col in col_iter:
            self.transformers[col] = _DistTransformer(self.t)
            self.transformers[col].fit(X[col])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out_df = X.copy()
        col_iter = tqdm(self._input_cols) if self.verbose else self._input_cols
        for col in col_iter:
            out_df[col] = self.transformers[col].transform(X[col])

        return out_df

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)
