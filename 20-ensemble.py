import logging

import hydra
import pandas as pd

import src.utils as utils
from src.ice_cube.scoring import angular_dist_score
from src.ice_cube.submission import to_submission_df

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    if c.settings.in_kaggle or not c.settings.is_training:
        c.settings.is_training = False

    utils.basic_environment_info()
    utils.fix_seed(utils.choice_seed(c))

    results = pd.read_csv("results.csv").set_index("event_id")
    results_stage2 = pd.read_csv("results_stage2.csv").set_index("event_id")

    try:
        assert (results.index == results_stage2.index).all()
        log.info("Results has same index.")
        results = (results + results_stage2) / 2.0
    except (AssertionError, ValueError):
        log.info("Results has different index.")
        results.loc[results_stage2.index, :] = (
            results[results.index.isin(results_stage2.index)] + results_stage2
        ) / 2.0

    submission_df = to_submission_df(results.reset_index())
    submission_df.to_csv("submission.csv")

    if c.settings.is_training:
        score = angular_dist_score(results["azimuth_y"], results["zenith_y"], results["azimuth_x"], results["zenith_x"])
        log.info(f"Base score: {score}")

        results_low_sigma = results[results["sigma"] <= c.inference_params.sigma_border]
        results_high_sigma = results[results["sigma"] > c.inference_params.sigma_border]
        log.info(
            f"Num of low sigma events: {len(results_low_sigma)}, Num of high sigma events: {len(results_high_sigma)}"
        )

        score_low_sigma = angular_dist_score(
            results_low_sigma["azimuth_y"],
            results_low_sigma["zenith_y"],
            results_low_sigma["azimuth_x"],
            results_low_sigma["zenith_x"],
        )
        score_high_sigma = angular_dist_score(
            results_high_sigma["azimuth_y"],
            results_high_sigma["zenith_y"],
            results_high_sigma["azimuth_x"],
            results_high_sigma["zenith_x"],
        )
        log.info(f"Low sigma score: {score_low_sigma}, High sigma score: {score_high_sigma}")

    log.info("Done.")


if __name__ == "__main__":
    main()
