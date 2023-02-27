# https://qiita.com/FukuharaYohei/items/7508f2146c63ffe16b1e

import logging
from typing import Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def preprocess(c, df: pd.DataFrame, stem: str) -> pd.DataFrame:
    # Convert None to NaN
    df = df.fillna(np.nan)

    ...

    # Convert None to NaN
    df = df.fillna(np.nan)

    return df


def preprocess_train_test(c, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat([train_df, test_df])
    log.info(f"Shape before preprocess: {df.shape}")

    return train_df, test_df
