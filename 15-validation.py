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
        log.info("Skip validation.")
        return

    utils.basic_environment_info()
    utils.fix_seed(utils.choice_seed(c))

    results = pd.read_csv("results.csv").set_index("event_id")
    submission_df = pd.read_csv("submission.csv").set_index("event_id")

    score = angular_dist_score(
        results["azimuth_y"], results["zenith_y"], submission_df["azimuth"], submission_df["zenith"]
    )
    log.info(f"Base score: {score}")

    submission_low_sigma = to_submission_df(results[results["sigma"] <= 0.5].reset_index())
    submission_high_sigma = to_submission_df(results[results["sigma"] > 0.5].reset_index())

    results_low_sigma = results[results["sigma"] <= 0.5]
    results_high_sigma = results[results["sigma"] > 0.5]

    score_low_sigma = angular_dist_score(
        results_low_sigma["azimuth_y"],
        results_low_sigma["zenith_y"],
        submission_low_sigma["azimuth"],
        submission_low_sigma["zenith"],
    )
    score_high_sigma = angular_dist_score(
        results_high_sigma["azimuth_y"],
        results_high_sigma["zenith_y"],
        submission_high_sigma["azimuth"],
        submission_high_sigma["zenith"],
    )
    log.info(
        f"Num of low sigma events: {len(submission_low_sigma)}, Num of high sigma events: {len(submission_high_sigma)}"
    )
    log.info(f"Low sigma score: {score_low_sigma}, High sigma score: {score_high_sigma}")

    log.info("Done.")


if __name__ == "__main__":
    main()
