import os

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.training.utils import make_train_validation_dataloader


def make_dataloader(c):
    """docstring for make_dataloader"""

    if c.settings.is_training:
        database_path = os.path.join(c.data.dir.working, "train_db.db")
    else:
        database_path = os.path.join(c.data.dir.working, "test_db.db")

    dataloader = make_train_validation_dataloader(
        db=database_path,
        selection=None,  # Entire database
        pulsemaps=c.data.ice_cube.pulse_table,
        features=FEATURES.KAGGLE,
        truth=TRUTH.KAGGLE,
        batch_size=c.training_params.batch_size,
        num_workers=c.training_params.num_workers,
        labels=None,  # Cannot make labels in test data
        index_column=c.settings.index_name,
        truth_table=c.data.ice_cube.meta_table,
        seed=c.global_params.seed,
    )

    return dataloader
