# https://qiita.com/FukuharaYohei/items/7508f2146c63ffe16b1e

import logging
from typing import Optional, Tuple

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def preprocess(c, df: pl.DataFrame, stem: str) -> pl.DataFrame:
    # Convert None to NaN
    # df = df.fill_null(np.nan)

    ...

    # Convert None to NaN
    # df = df.fill_null(np.nan)

    return df


def preprocess_train_test(c, train_df: pl.DataFrame, test_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    df = pl.concat([train_df, test_df])
    log.info(f"Shape before preprocess: {df.shape}")

    return train_df, test_df
