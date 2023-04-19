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
from src.ice_cube.data_loader import CollateFn, make_dataloader_batch
from src.ice_cube.model import load_pretrained_model
from src.ice_cube.submission import to_submission_df
from src.preprocess import preprocess

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    if c.settings.in_kaggle:
        c.settings.is_training = False
        c.data.dir.pretrained = "/kaggle/input"

    utils.basic_environment_info()
    utils.fix_seed(utils.choice_seed(c))

    if c.training_params.stage2:
        log.info("Stage2 inference.")
        results_stage1 = pd.read_csv("results.csv").set_index("event_id")
        results_stage1 = results_stage1[results_stage1["sigma"] > c.inference_params.sigma_border]
        c.global_params.seed += 1
        utils.fix_seed(utils.choice_seed(c))

    if c.settings.is_training:
        metadata_path = os.path.join(c.data.dir.input, "train_meta.parquet")
        input_batch_dir = c.data.dir.input_train
    else:
        metadata_path = os.path.join(c.data.dir.input, "test_meta.parquet")
        input_batch_dir = c.data.dir.input_test

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

        batch_df = pd.read_parquet(path=f"{input_batch_dir}/batch_{batch_id[0]}.parquet").reset_index()
        batch_df = pd.merge(batch_df, sensor_df, on="sensor_id").sort_values("event_id")
        batch_df = preprocess(c, batch_df, "batch")

        if c.training_params.stage2:
            dataloader = make_dataloader_batch(c, meta_df, batch_df, CollateFn(), results_stage1.index)
            model = load_pretrained_model(c, dataloader, state_dict_path=c.inference_params.model_path_stage2)
        else:
            log.info(f"Pulse limit: {c.inference_params.pulse_limit}")
            dataloader = make_dataloader_batch(
                c, meta_df, batch_df, CollateFn(pulse_limit=c.inference_params.pulse_limit)
            )
            model = load_pretrained_model(c, dataloader, state_dict_path=c.inference_params.model_path)

        log.info("Predict by default features with pulse cut.")
        results = model.predict(gpus=[0], dataloader=dataloader)
        results = torch.cat(results, dim=1).detach().cpu().numpy()
        if c.inference_params.kappa_weight:
            results[:, 0:3] *= np.sqrt(results[:, 3]).reshape(-1, 1)
        n_count = 1

        log.info("Predict by default features.")
        dataloader = make_dataloader_batch(c, meta_df, batch_df, CollateFn(pulse_limit=None))
        results_ = model.predict(gpus=[0], dataloader=dataloader)
        results_ = torch.cat(results_, dim=1).detach().cpu().numpy()
        if c.inference_params.kappa_weight:
            results_[:, 0:3] *= np.sqrt(results_[:, 3]).reshape(-1, 1)
        results += results_
        n_count += 1

        if c.inference_params.n_ensemble > 1:
            utils.fix_seed(c.global_params.seed + 180)
            log.info("Predict by features that invert x and y with pulse cut.")
            if c.training_params.stage2:
                dataloader = make_dataloader_batch(c, meta_df, batch_df, CollateFn(x=-1, y=-1), results_stage1.index)
            else:
                dataloader = make_dataloader_batch(
                    c, meta_df, batch_df, CollateFn(x=-1, y=-1, pulse_limit=c.inference_params.pulse_limit)
                )
            results_ = model.predict(gpus=[0], dataloader=dataloader)
            results_ = torch.cat(results_, dim=1).detach().cpu().numpy()
            results_[:, 0] *= -1
            results_[:, 1] *= -1
            if c.inference_params.kappa_weight:
                results_[:, 0:3] *= np.sqrt(results_[:, 3]).reshape(-1, 1)
            results += results_
            n_count += 1

            log.info("Predict by features that invert x and y.")
            dataloader = make_dataloader_batch(c, meta_df, batch_df, CollateFn(x=-1, y=-1, pulse_limit=None))
            results_ = model.predict(gpus=[0], dataloader=dataloader)
            results_ = torch.cat(results_, dim=1).detach().cpu().numpy()
            results_[:, 0] *= -1
            results_[:, 1] *= -1
            if c.inference_params.kappa_weight:
                results_[:, 0:3] *= np.sqrt(results_[:, 3]).reshape(-1, 1)
            results += results_
            n_count += 1

        if c.inference_params.n_ensemble > 2:
            utils.fix_seed(c.global_params.seed + 90)
            log.info("Predict by features that invert x with pulse cut.")
            if c.training_params.stage2:
                dataloader = make_dataloader_batch(c, meta_df, batch_df, CollateFn(x=-1), results_stage1.index)
            else:
                dataloader = make_dataloader_batch(
                    c, meta_df, batch_df, CollateFn(x=-1, pulse_limit=c.inference_params.pulse_limit)
                )
            results_ = model.predict(gpus=[0], dataloader=dataloader)
            results_ = torch.cat(results_, dim=1).detach().cpu().numpy()
            results_[:, 0] *= -1
            if c.inference_params.kappa_weight:
                results_[:, 0:3] *= np.sqrt(results_[:, 3]).reshape(-1, 1)
            results += results_
            n_count += 1

            # log.info("Predict by features that invert x.")
            # dataloader = make_dataloader_batch(c, meta_df, batch_df, CollateFn(x=-1, pulse_limit=None))
            # results_ = model.predict(gpus=[0], dataloader=dataloader)
            # results_ = torch.cat(results_, dim=1).detach().cpu().numpy()
            # results_[:, 0] *= -1
            # results_[:, 0:3] *= np.sqrt(results_[:, 3]).reshape(-1, 1)
            # results += results_
            # n_count += 1

            utils.fix_seed(c.global_params.seed + 270)
            log.info("Predict by features that invert y with pulse cut.")
            if c.training_params.stage2:
                dataloader = make_dataloader_batch(c, meta_df, batch_df, CollateFn(y=-1), results_stage1.index)
            else:
                dataloader = make_dataloader_batch(
                    c, meta_df, batch_df, CollateFn(y=-1, pulse_limit=c.inference_params.pulse_limit)
                )
            results_ = model.predict(gpus=[0], dataloader=dataloader)
            results_ = torch.cat(results_, dim=1).detach().cpu().numpy()
            results_[:, 1] *= -1
            if c.inference_params.kappa_weight:
                results_[:, 0:3] *= np.sqrt(results_[:, 3]).reshape(-1, 1)
            results += results_
            n_count += 1

            # log.info("Predict by features that invert y.")
            # dataloader = make_dataloader_batch(c, meta_df, batch_df, CollateFn(y=-1, pulse_limit=None))
            # results_ = model.predict(gpus=[0], dataloader=dataloader)
            # results_ = torch.cat(results_, dim=1).detach().cpu().numpy()
            # results_[:, 1] *= -1
            # results_[:, 0:3] *= np.sqrt(results_[:, 3]).reshape(-1, 1)
            # results += results_
            # n_count += 1

        log.info(f"Num of ensemble: {n_count}.")
        results = results / n_count

        predictions.append(results)
        event_ids.append(dataloader.dataset.event_ids)

        if c.settings.is_training:
            validations_df.append(meta_df[["event_id", "azimuth", "zenith"]])

    event_ids = np.concatenate(event_ids)
    predictions = np.concatenate(predictions)

    predictions_df = pd.DataFrame(event_ids, columns=["event_id"])
    if "direction" in c.model_params.tasks:
        predictions_df[["direction_x", "direction_y", "direction_z", "direction_kappa"]] = predictions
    else:
        predictions_df[["azimuth", "azimuth_kappa", "zenith", "zenith_kappa"]] = predictions
        predictions_df["direction_kappa"] = (predictions_df["azimuth_kappa"] + predictions_df["zenith_kappa"]) / 2.0

    log.info("Make submission.")
    submission_df = to_submission_df(predictions_df)
    if c.training_params.stage2:
        submission_df.to_csv("submission_stage2.csv")
    else:
        submission_df.to_csv("submission.csv")

    if c.settings.is_training:
        validations_df = pd.concat(validations_df)
        predictions_df.loc[:, "sigma"] = 1 / np.sqrt(predictions_df["direction_kappa"])
        predictions_df = pd.merge(predictions_df, validations_df, on=["event_id"]).sort_values("event_id")

    if c.training_params.stage2:
        predictions_df.to_csv("results_stage2.csv")
    else:
        predictions_df.to_csv("results.csv")

    log.info(f"results columns: {predictions_df.columns}")
    log.info(f"submission columns: {submission_df.columns}")

    log.info("Done.")


if __name__ == "__main__":
    main()
