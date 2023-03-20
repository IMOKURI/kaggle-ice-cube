import logging
import os

import hydra
from graphnet.training.callbacks import ProgressBar
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import EarlyStopping

import src.utils as utils
from src.ice_cube.data_loader import make_train_dataloader
from src.ice_cube.model import build_model

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

    database_path = os.path.join(c.data.dir.dataset, f"train_{c.data.ice_cube.train_batch}_db.db")
    train_loader, valid_loader = make_train_dataloader(c, database_path)

    model = build_model(c, train_loader, custom_aggregation=True)

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
        gpus=[0, 1, 2],
        # distribution_strategy="ddp",
        max_epochs=c.training_params.epoch,
    )

    model.save_state_dict(os.path.join(run_dir, "state_dict.pth"))
    model.save(os.path.join(run_dir, "model.pth"))

    log.info("Done.")


if __name__ == "__main__":
    main()
