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

    train_batch = list(
        range(c.data.ice_cube.train_batch, c.data.ice_cube.train_batch + c.data.ice_cube.train_batch_size)
    )

    event_ids = []
    predictions = []
    validations_df = []
    for n, meta_df in enumerate(metadata_iter):
        if c.settings.is_training and n not in train_batch:
            continue

        meta_df = meta_df.to_pandas()

        batch_id = pd.unique(meta_df["batch_id"])
        assert len(batch_id) == 1, "contains multiple batch_ids. Did you set the batch_size correctly?"
        log.info(f"Batch {batch_id} ...")

        dataloader = make_test_dataloader_batch(c, batch_id[0], meta_df, sensor_df, collate_fn)
        model = load_pretrained_model(c, dataloader, state_dict_path=c.inference_params.model_path)

        log.info("Predict by default features.")
        results = model.predict(gpus=[0], dataloader=dataloader)
        results_plus_plus = torch.cat(results, dim=1).detach().cpu().numpy()

        if c.settings.is_training:
            n_ensemble = 1
        else:
            n_ensemble = 4

        if n_ensemble > 1:
            log.info("Predict by features that invert x and y.")
            dataloader = make_test_dataloader_batch(c, batch_id[0], meta_df, sensor_df, collate_fn_minus_minus)
            results = model.predict(gpus=[0], dataloader=dataloader)
            results_minus_minus = torch.cat(results, dim=1).detach().cpu().numpy()
            results_minus_minus[:, 0] *= -1
            results_minus_minus[:, 1] *= -1

            if n_ensemble > 2:
                log.info("Predict by features that invert x.")
                dataloader = make_test_dataloader_batch(c, batch_id[0], meta_df, sensor_df, collate_fn_minus_plus)
                results = model.predict(gpus=[0], dataloader=dataloader)
                results_minus_plus = torch.cat(results, dim=1).detach().cpu().numpy()
                results_minus_plus[:, 0] *= -1

                log.info("Predict by features that invert y.")
                dataloader = make_test_dataloader_batch(c, batch_id[0], meta_df, sensor_df, collate_fn_plus_minus)
                results = model.predict(gpus=[0], dataloader=dataloader)
                results_plus_minus = torch.cat(results, dim=1).detach().cpu().numpy()
                results_plus_minus[:, 1] *= -1

                log.info("Ensemble 4.")
                results = (results_plus_plus + results_minus_minus + results_minus_plus + results_plus_minus) / 4.0

            else:
                log.info("Ensemble 2.")
                results = (results_plus_plus + results_minus_minus) / 2.0

        else:
            results = results_plus_plus

        predictions.append(results)
        event_ids.append(dataloader.dataset.event_ids)

        if c.settings.is_training:
            validations_df.append(meta_df[["event_id", "azimuth", "zenith"]])

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
