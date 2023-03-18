"""
Parquetファイルからデータを読み込んで推論
"""

import logging
import os

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import src.utils as utils
from src.ice_cube.data_loader import (
    collate_fn,
    collate_fn_minus_minus,
    collate_fn_minus_plus,
    collate_fn_plus_minus,
    make_test_dataloader_batch,
)
from src.ice_cube.model import load_pretrained_model
from src.ice_cube.submission import to_submission_df

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    if c.settings.in_kaggle:
        c.settings.is_training = False
        c.data.dir.pretrained = "/kaggle/input"

    utils.basic_environment_info()
    utils.fix_seed(utils.choice_seed(c))

    if c.settings.is_training:
        metadata_path = os.path.join(c.data.dir.input, "train_meta.parquet")
    else:
        metadata_path = os.path.join(c.data.dir.input, "test_meta.parquet")

    sensor_df = pd.read_csv(os.path.join(c.data.dir.input, "sensor_geometry.csv"))

    batch_size = 200_000
    metadata_iter = pq.ParquetFile(metadata_path).iter_batches(batch_size=batch_size)

    event_ids = []
    predictions = []
    validations_df = []
    for n, meta_df in enumerate(metadata_iter):
        meta_df = meta_df.to_pandas()

        batch_id = pd.unique(meta_df["batch_id"])
        assert len(batch_id) == 1, "contains multiple batch_ids. Did you set the batch_size correctly?"
        log.info(f"Batch {batch_id} ...")

        dataloader = make_test_dataloader_batch(c, batch_id[0], meta_df, sensor_df, collate_fn)
        model = load_pretrained_model(c, dataloader, state_dict_path=c.inference_params.model_path)

        log.info("Predict by default features.")
        results = model.predict(gpus=[0], dataloader=dataloader)
        results = torch.cat(results, dim=1).detach().cpu().numpy()

        predictions.append(results)
        event_ids.append(dataloader.dataset.event_ids)

        if c.settings.is_training:
            validations_df.append(meta_df[["event_id", "azimuth", "zenith"]])

            if n + 1 >= c.data.ice_cube.train_batch_size:
                break

    event_ids = np.concatenate(event_ids)
    predictions = np.concatenate(predictions)

    predictions_df = pd.DataFrame(event_ids, columns=["event_id"])
    predictions_df[["direction_x", "direction_y", "direction_z", "direction_kappa"]] = predictions

    log.info("Make submission.")
    submission_df = to_submission_df(predictions_df)
    submission_df.to_csv("submission.csv")

    if c.settings.is_training:
        validations_df = pd.concat(validations_df)
        predictions_df.loc[:, "sigma"] = 1 / np.sqrt(predictions_df["direction_kappa"])
        predictions_df = pd.merge(predictions_df, validations_df, on=["event_id"])
        predictions_df.to_csv("results.csv")

    log.info("Done.")


if __name__ == "__main__":
    main()
