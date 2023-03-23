"""
線形回帰で推論
"""

import logging
import os

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

import src.utils as utils
from src.ice_cube.linear_regression import compute_angle_numba
from src.ice_cube.submission import to_submission_df

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    if c.settings.in_kaggle:
        c.settings.is_training = False
        c.data.dir.pretrained = "/kaggle/input"

    utils.basic_environment_info()
    utils.fix_seed(utils.choice_seed(c))

    # results = pd.read_parquet("results_low_sigma.parquet")
    results = pd.read_parquet("results_high_sigma.parquet")

    if c.settings.is_training:
        metadata_path = os.path.join(c.data.dir.input, "train_meta.parquet")
    else:
        metadata_path = os.path.join(c.data.dir.input, "test_meta.parquet")

    sensor_df = pd.read_csv(os.path.join(c.data.dir.input, "sensor_geometry.csv"))

    batch_size = 200_000
    metadata_iter = pq.ParquetFile(metadata_path).iter_batches(batch_size=batch_size)

    train_batch = list(
        range(c.data.ice_cube.train_batch, c.data.ice_cube.train_batch + c.data.ice_cube.train_batch_size)
    )

    if c.settings.is_training:
        input_batch_dir = c.data.dir.input_train
    else:
        input_batch_dir = c.data.dir.input_test

    event_ids = []
    azimuth_preds = []
    zenith_preds = []
    validations_df = []
    for n, meta_df in enumerate(metadata_iter):
        if c.settings.is_training and n not in train_batch:
            continue

        meta_df = meta_df.to_pandas()

        batch_id = pd.unique(meta_df["batch_id"])
        assert len(batch_id) == 1, "contains multiple batch_ids. Did you set the batch_size correctly?"
        log.info(f"Batch {batch_id} ...")

        batch_df = pd.read_parquet(path=f"{input_batch_dir}/batch_{batch_id[0]}.parquet").reset_index()

        meta_df = meta_df[meta_df["event_id"].isin(results.index)]

        for event_id in tqdm(meta_df["event_id"]):
            first_index = meta_df[meta_df["event_id"] == event_id]["first_pulse_index"].to_numpy()[0]
            last_index = meta_df[meta_df["event_id"] == event_id]["last_pulse_index"].to_numpy()[0]
            event = batch_df.iloc[first_index:last_index, :]

            pulse_limit = 300
            if len(event) > pulse_limit:
                event = event.loc[np.random.choice(event.index, pulse_limit), :]

            position = sensor_df.loc[event["sensor_id"]].to_numpy()
            pred_azimuth, pred_zenith = compute_angle_numba(position[:, 0], position[:, 1], position[:, 2])

            event_ids.append(event_id)
            azimuth_preds.append(pred_azimuth)
            zenith_preds.append(pred_zenith)

        if c.settings.is_training:
            validations_df.append(meta_df[["event_id", "azimuth", "zenith"]])

    predictions_df = pd.DataFrame(event_ids, columns=["event_id"])
    predictions_df["azimuth"] = azimuth_preds
    predictions_df["zenith"] = zenith_preds

    log.info("Make submission.")
    submission_df = to_submission_df(predictions_df)
    submission_df.to_csv("submission_linear_regression.csv")

    if c.settings.is_training:
        validations_df = pd.concat(validations_df)
        predictions_df = pd.merge(predictions_df, validations_df, on=["event_id"]).sort_values("event_id")
        predictions_df.to_csv("results_linear_regression.csv")

    log.info(f"results columns: {predictions_df.columns}")
    log.info(f"submission columns: {submission_df.columns}")

    log.info("Done.")


if __name__ == "__main__":
    main()
