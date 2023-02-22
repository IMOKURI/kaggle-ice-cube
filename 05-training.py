import logging
import os

import hydra
import pandas as pd
from graphnet.training.callbacks import ProgressBar
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import EarlyStopping

import src.utils as utils
from src.ice_cube.data_loader import make_train_dataloader
from src.ice_cube.model import build_model
from src.ice_cube.scoring import angular_dist_score
from src.ice_cube.submission import to_submission_df

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    if c.settings.in_kaggle:
        log.warning("Training should not be done on kaggle.")
        return

    utils.basic_environment_info()
    utils.fix_seed(utils.choice_seed(c))

    run_dir = HydraConfig.get().run.dir
    log.info(f"Run dir: {run_dir}")

    train_loader, valid_loader = make_train_dataloader(c)

    model = build_model(c, train_loader)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=c.training_params.es_patience,
        ),
        ProgressBar(),
    ]

    model.fit(
        train_loader,
        valid_loader,
        callbacks=callbacks,
        gpus=[0],
        # distribution_strategy="ddp",
        max_epochs=c.training_params.epoch,
    )

    results = model.predict_as_dataframe(
        gpus=[0],
        dataloader=valid_loader,
        prediction_columns=model.prediction_columns,
        additional_attributes=[c.settings.index_name],
    )

    submission_df = to_submission_df(results)
    submission_df.to_csv(os.path.join(run_dir, "oof_pred.csv"))

    valid_data = valid_loader.dataset.query_table("meta_table", ["event_id", "azimuth", "zenith"])
    valid_df = (
        pd.DataFrame(valid_data, columns=["event_id", "azimuth", "zenith"])
        .set_index("event_id")
        .loc[submission_df.index, :]
    )
    score = angular_dist_score(
        valid_df["azimuth"], valid_df["zenith"], submission_df["azimuth"], submission_df["zenith"]
    )
    log.info(f"score: {score}")
    valid_df.to_csv(os.path.join(run_dir, "oof_true.csv"))

    model.save_state_dict(os.path.join(run_dir, "state_dict.pth"))
    model.save(os.path.join(run_dir, "model.pth"))

    log.info("Done.")


if __name__ == "__main__":
    main()
