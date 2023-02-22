import os

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.training.labels import Direction
from graphnet.training.utils import make_dataloader, make_train_validation_dataloader


def make_train_dataloader(c):
    database_path = os.path.join(c.data.dir.dataset, f"train_{c.data.ice_cube.train_batch}_db.db")

    train_loader, valid_loader = make_train_validation_dataloader(
        db=database_path,
        selection=None,  # Entire database
        pulsemaps=c.data.ice_cube.pulse_table,
        features=FEATURES.KAGGLE,
        truth=TRUTH.KAGGLE,
        labels={"direction": Direction()},
        batch_size=c.training_params.batch_size,
        # num_workers=c.training_params.num_workers,
        index_column=c.settings.index_name,
        truth_table=c.data.ice_cube.meta_table,
        seed=c.global_params.seed,
        test_size=0.25,
    )

    return train_loader, valid_loader


def make_test_dataloader(c):
    database_path = os.path.join(c.data.dir.dataset, "test_db.db")

    dataloader = make_dataloader(
        db=database_path,
        selection=None,  # Entire database
        pulsemaps=c.data.ice_cube.pulse_table,
        features=FEATURES.KAGGLE,
        truth=TRUTH.KAGGLE,
        labels={"direction": Direction()},
        batch_size=c.training_params.batch_size,
        shuffle=False,
        # num_workers=c.training_params.num_workers,
        index_column=c.settings.index_name,
        truth_table=c.data.ice_cube.meta_table,
    )

    return dataloader
