import logging

import hydra
import pandas as pd

import src.utils as utils
from src.ice_cube.submission import to_submission_df

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    if c.settings.in_kaggle:
        ...

    utils.basic_environment_info()
    utils.fix_seed(utils.choice_seed(c))

    log.info("Read results_default.csv")
    results = pd.read_csv("results_default.csv").set_index("event_id")

    log.info("Read results_minus_minus.csv")
    results_minus_minus = pd.read_csv("results_minus_minus.csv").set_index("event_id")

    results = results + results_minus_minus

    results = results / 2
    results.to_csv("results.csv")

    submission_df = to_submission_df(results.reset_index())
    submission_df.to_csv("submission.csv")

    log.info("Done.")


if __name__ == "__main__":
    main()
