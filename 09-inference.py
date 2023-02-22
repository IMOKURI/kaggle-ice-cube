import logging

import hydra
import pandas as pd

import src.utils as utils
from src.ice_cube.data_loader import make_test_dataloader, make_train_dataloader
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

    if c.settings.is_training:
        _, dataloader = make_train_dataloader(c)
    else:
        dataloader = make_test_dataloader(c)

    model = load_pretrained_model(c, dataloader)
    results = model.predict_as_dataframe(
        gpus=[0],
        dataloader=dataloader,
        prediction_columns=model.prediction_columns,
        additional_attributes=[c.settings.index_name],
    )

    submission_df = to_submission_df(results)
    submission_df.to_csv("submission.csv")

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
        log.info(f"score: {score}")
        valid_df.to_csv("valid.csv")

    log.info("Done.")


if __name__ == "__main__":
    main()
