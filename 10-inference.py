import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch

import src.utils as utils
from src.ice_cube.data_loader import (
    collate_fn_minus_minus,
    collate_fn_minus_plus,
    collate_fn_plus_minus,
    make_test_dataloader,
    make_test_dataloader_custom,
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
        database_path = os.path.join(c.data.dir.dataset, f"train_{c.data.ice_cube.train_batch}_db.db")
    else:
        database_path = os.path.join(c.data.dir.dataset, "test_db.db")
    log.info(f"Database path: {database_path}")

    dataloader = make_test_dataloader(c, database_path)

    model = load_pretrained_model(c, dataloader, state_dict_path=c.inference_params.model_path)

    log.info("Predict by default features.")
    results = model.predict(gpus=[0], dataloader=dataloader)
    predictions = torch.cat(results, dim=1).detach().cpu().numpy()

    log.info("Predict by features that invert x and y.")
    dataloader = make_test_dataloader_custom(c, database_path, collate_fn_minus_minus)
    results = model.predict(gpus=[0], dataloader=dataloader)
    predictions_minus_minus = torch.cat(results, dim=1).detach().cpu().numpy()
    predictions_minus_minus[:, 0] *= -1
    predictions_minus_minus[:, 1] *= -1

    # log.info("Predict by features that invert x.")
    # dataloader = make_test_dataloader_custom(c, database_path, collate_fn_minus_plus)
    # results = model.predict(gpus=[0], dataloader=dataloader)
    # predictions_minus_plus = torch.cat(results, dim=1).detach().cpu().numpy()
    # predictions_minus_plus[:, 0] *= -1

    # log.info("Predict by features that invert y.")
    # dataloader = make_test_dataloader_custom(c, database_path, collate_fn_plus_minus)
    # results = model.predict(gpus=[0], dataloader=dataloader)
    # predictions_plus_minus = torch.cat(results, dim=1).detach().cpu().numpy()
    # predictions_plus_minus[:, 1] *= -1

    log.info("Ensemble.")
    predictions = (predictions + predictions_minus_minus) / 2.0
    # predictions = (predictions + predictions_minus_minus + predictions_minus_plus + predictions_plus_minus) / 4.0

    predictions_df = pd.DataFrame(dataloader.dataset.query_table("meta_table", ["event_id"]), columns=["event_id"])
    predictions_df[["direction_x", "direction_y", "direction_z", "direction_kappa"]] = predictions

    log.info("Make submission.")
    submission_df = to_submission_df(predictions_df)
    submission_df.to_csv("submission.csv")

    if c.settings.is_training:
        predictions_df.loc[:, "sigma"] = 1 / np.sqrt(predictions_df["direction_kappa"])
        predictions_df.to_csv("results.csv")

        valid_data = dataloader.dataset.query_table("meta_table", ["event_id", "azimuth", "zenith"])
        valid_df = pd.DataFrame(valid_data, columns=["event_id", "azimuth", "zenith"]).set_index("event_id")
        valid_df.to_csv("valid.csv")

    log.info("Done.")


if __name__ == "__main__":
    main()
