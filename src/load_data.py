import logging
import os

import pandas as pd

from .make_fold import make_fold
from .preprocess import preprocess, preprocess_train_test

log = logging.getLogger(__name__)


class LoadData:
    def __init__(self, c, use_fold=True, do_preprocess=True):
        self.c = c

        for file_name in c.data.input:
            stem = os.path.splitext(file_name)[0].replace("/", "__")
            extension = os.path.splitext(file_name)[1]

            original_file_path = os.path.join(c.data.dir.input, file_name)

            if os.path.exists(original_file_path):
                log.info(f"Load original file. path: {original_file_path}")

                if extension == ".csv":
                    df = pd.read_csv(original_file_path)

                elif extension == ".parquet":
                    df = pd.read_parquet(original_file_path)

                else:
                    raise Exception(f"Invalid extension to load file. filename: {original_file_path}")

            else:
                log.warning(f"File does not exist. path: {original_file_path}")
                continue

            # if c.settings.debug:
            #     df = sample_for_debug(c, df)

            if stem in ["train_meta", "test_meta"] and do_preprocess:
                df = preprocess(c, df, stem)

            setattr(self, stem, df)

        if do_preprocess:
            train = getattr(self, f"train_meta")
            test = getattr(self, f"test_meta")

            train, test = preprocess_train_test(c, train, test)

            setattr(self, "train_meta", train)
            setattr(self, "test_meta", test)

        if use_fold:
            train = getattr(self, f"train_meta")
            test = getattr(self, f"test_meta")

            train = make_fold(c, train)
            test = make_fold(c, test)

            setattr(self, "train_meta", train)
            setattr(self, "test_meta", test)


def sample_for_debug(c, df):
    if len(df) > c.settings.n_debug_data and c.settings.n_debug_data > 0:
        df = df.sample(n=c.settings.n_debug_data, random_state=c.global_params.seed).reset_index(drop=True)
    return df
