import logging
import os

import hydra
import numpy as np
import pandas as pd

import src.utils as utils
from src.ice_cube.data_loader import make_test_dataloader
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

    results = model.predict_as_dataframe(
        gpus=[0],
        dataloader=dataloader,
        prediction_columns=model.prediction_columns,
        additional_attributes=[c.settings.index_name],
    )

    results.loc[:, "sigma"] = 1 / np.sqrt(results["direction_kappa"])
    results.to_csv("results.csv")

    submission_df = to_submission_df(results)
    submission_df.to_csv("submission.csv")

    if c.settings.is_training:
        valid_data = dataloader.dataset.query_table("meta_table", ["event_id", "azimuth", "zenith"])
        valid_df = (
            pd.DataFrame(valid_data, columns=["event_id", "azimuth", "zenith"])
            .set_index("event_id")
            .loc[submission_df.index, :]
        )
        valid_df.to_csv("valid.csv")

    log.info("Done.")


if __name__ == "__main__":
    main()
