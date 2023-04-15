# https://qiita.com/FukuharaYohei/items/7508f2146c63ffe16b1e

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from .preprocesses.cache import transform_data
from .preprocesses.p001_dist_transformer import DistTransformer

log = logging.getLogger(__name__)


def preprocess(c, df: pd.DataFrame, stem: str) -> pd.DataFrame:
    # Convert None to NaN
    # df = df.fillna(np.nan)

    if c.model_params.detector == "custom":
        pp = DistTransformer(transform="min-max", verbose=True)
        df.loc[:, ["x", "y", "z", "time"]] = pp.fit_transform(df.loc[:, ["x", "y", "z", "time"]])

        pp = DistTransformer(transform="yeo-johnson", verbose=True)
        df.loc[:, ["charge"]] = pp.fit_transform(df.loc[:, ["charge"]])

    if c.training_params.stage2:
        df.loc[:, ["auxiliary"]] = df["auxiliary"].replace({True: 1, False: 1})
    else:
        df.loc[:, ["auxiliary"]] = df["auxiliary"].replace({True: 1, False: 0})

    # Convert None to NaN
    # df = df.fillna(np.nan)

    return df


def preprocess_train_test(c, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat([train_df, test_df])
    log.info(f"Shape before preprocess: {df.shape}")

    return train_df, test_df
