import logging
import os
import pickle
import re
from functools import wraps
from typing import Callable, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def load_or_fit(func: Callable):
    """
    前処理を行うクラスがすでに保存されていれば、それをロードする。
    保存されていなければ、 func で生成、学習する。
    与えられたデータを、学習済みクラスで前処理する。

    Args:
        func (Callable): 前処理を行うクラスのインスタンスを生成し、学習する関数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        c = args[0]
        path = os.path.join(c.data.dir.preprocess, args[1]) if args[1] is not None else None

        if path is not None and os.path.exists(path):
            instance = pickle.load(open(path, "rb"))

        else:
            instance = func(*args, **kwargs)

            if path is not None:
                os.makedirs(c.data.dir.preprocess, exist_ok=True)
                pickle.dump(instance, open(path, "wb"), protocol=4)

        return instance

    return wrapper


def load_or_transform(func: Callable):
    """
    前処理されたデータがすでに存在すれば、それをロードする。
    存在しなければ、 func で生成する。生成したデータは保存しておく。

    Args:
        func (Callable): 前処理を行う関数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        c = args[0]
        path = os.path.join(c.data.dir.preprocess, args[1])

        if os.path.exists(path) and os.path.splitext(path)[1] == ".npy":
            array = np.load(path, allow_pickle=True)
        elif os.path.exists(path) and os.path.splitext(path)[1] == ".pickle":
            array = pd.read_pickle(path)

        else:
            array = func(*args, **kwargs)

            if isinstance(array, np.ndarray):
                os.makedirs(c.data.dir.preprocess, exist_ok=True)
                np.save(os.path.splitext(path)[0], array)
            elif isinstance(array, pd.DataFrame):
                os.makedirs(c.data.dir.preprocess, exist_ok=True)
                array.to_pickle(path)

        return array

    return wrapper


@load_or_fit
def fit_instance(_, path, data: np.ndarray, instance, label=None):
    if label is None:
        instance.fit(data)
    else:
        instance.fit(data, label)

    log.info(f"Fit preprocess. -> {path}")
    return instance


@load_or_transform
def transform_data(
    c, path, data: Union[np.ndarray, pd.DataFrame], instance, label=None
) -> Union[np.ndarray, pd.DataFrame]:
    instance = fit_instance(
        c, re.sub("\w+-", "", path).replace(".npy", ".pkl").replace(".pickle", ".pkl"), data, instance, label
    )
    features = instance.transform(data)

    log.info(f"Transform data. -> {path}, shape: {features.shape}")
    return features
