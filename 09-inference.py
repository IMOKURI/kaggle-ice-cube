import logging

import hydra
import numpy as np
import pandas as pd

import src.utils as utils
from src.ice_cube.data_loader import make_test_dataloader, make_test_dataloader_plus_minus, make_train_dataloader
from src.ice_cube.model import load_pretrained_model
from src.ice_cube.scoring import angular_dist_score
from src.ice_cube.submission import to_submission_df

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    if c.settings.in_kaggle:
        c.settings.is_training = False
        c.data.dir.pretrained = "/kaggle/input/dynedge-pretrained"

    utils.basic_environment_info()
    utils.fix_seed(utils.choice_seed(c))

    dataloader = make_test_dataloader(c)
    model = load_pretrained_model(c, dataloader)

    results = model.predict_as_dataframe(
        gpus=[0],
        dataloader=dataloader,
        prediction_columns=model.prediction_columns,
        additional_attributes=[c.settings.index_name],
    )

    submission_df = to_submission_df(results)
    # submission_df.to_csv("submission.csv")

    if c.inference_params.tta:
        log.info(f"Enable TTA.")
        dataloader = make_test_dataloader_plus_minus(c)
        results_tta = model.predict_as_dataframe(
            gpus=[0],
            dataloader=dataloader,
            prediction_columns=model.prediction_columns,
            additional_attributes=[c.settings.index_name],
        )

        results_tta["direction_x"] = -results_tta["direction_x"]
        results_tta["direction_y"] = -results_tta["direction_y"]
        results_tta["direction_z"] = -results_tta["direction_z"]

        submission_tta_df = to_submission_df(results_tta)
        # submission_tta_df.to_csv("submission_tta.csv")

    if c.settings.is_training:
        valid_data = dataloader.dataset.query_table("meta_table", ["event_id", "azimuth", "zenith"])
        valid_df = (
            pd.DataFrame(valid_data, columns=["event_id", "azimuth", "zenith"])
            .set_index("event_id")
            .loc[submission_df.index, :]
        )
        score = angular_dist_score(
            valid_df["azimuth"], valid_df["zenith"], submission_df["azimuth"], submission_df["zenith"]
        )
        log.info(f"Base score: {score}")
        # valid_df.to_csv("valid.csv")

        if c.inference_params.tta:
            score = angular_dist_score(
                valid_df["azimuth"], valid_df["zenith"], submission_tta_df["azimuth"], submission_tta_df["zenith"]
            )
            log.info(f"TTA score: {score}")

    if c.inference_params.tta:
        results = (results + results_tta) / 2
        submission_df = to_submission_df(results)

        if c.settings.is_training:
            score = angular_dist_score(
                valid_df["azimuth"], valid_df["zenith"], submission_df["azimuth"], submission_df["zenith"]
            )
            log.info(f"Final score: {score}")

    if c.settings.is_training:
        results.loc[:, "sigma"] = 1 / np.sqrt(results["direction_kappa"])

        submission_low_sigma = to_submission_df(results[results["sigma"] <= 0.5])
        submission_high_sigma = to_submission_df(results[results["sigma"] > 0.5])

        results.set_index("event_id", inplace=True)
        valid_low_sigma = valid_df[results["sigma"] <= 0.5]
        valid_high_sigma = valid_df[results["sigma"] > 0.5]

        score_low_sigma = angular_dist_score(
            valid_low_sigma["azimuth"],
            valid_low_sigma["zenith"],
            submission_low_sigma["azimuth"],
            submission_low_sigma["zenith"],
        )
        score_high_sigma = angular_dist_score(
            valid_high_sigma["azimuth"],
            valid_high_sigma["zenith"],
            submission_high_sigma["azimuth"],
            submission_high_sigma["zenith"],
        )
        log.info(f"Low sigma score: {score_low_sigma}, High sigma score: {score_high_sigma}")

    submission_df.to_csv("submission.csv")
    log.info("Done.")


if __name__ == "__main__":
    main()
