import os
from typing import List

import torch
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite import SQLiteDataset
from graphnet.training.labels import Direction
from graphnet.training.utils import make_dataloader, make_train_validation_dataloader
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data


def make_train_dataloader(c, selection=None):
    database_path = os.path.join(c.data.dir.dataset, f"train_{c.data.ice_cube.train_batch}_db.db")

    train_loader, valid_loader = make_train_validation_dataloader(
        db=database_path,
        selection=selection,  # Entire database
        pulsemaps=c.data.ice_cube.pulse_table,
        features=FEATURES.KAGGLE,
        truth=TRUTH.KAGGLE,
        labels={"direction": Direction()},
        batch_size=c.training_params.batch_size,
        num_workers=c.training_params.num_workers,
        index_column=c.settings.index_name,
        truth_table=c.data.ice_cube.meta_table,
        seed=c.global_params.seed,
        test_size=0.25,
    )

    return train_loader, valid_loader


def make_test_dataloader(c, selection=None):
    if c.settings.is_training:
        database_path = os.path.join(c.data.dir.dataset, f"train_{c.data.ice_cube.train_batch}_db.db")
    else:
        database_path = os.path.join(c.data.dir.dataset, "test_db.db")

    dataloader = make_dataloader(
        db=database_path,
        selection=selection,  # Entire database: None
        pulsemaps=c.data.ice_cube.pulse_table,
        features=FEATURES.KAGGLE,
        truth=TRUTH.KAGGLE,
        # labels={"direction": Direction()},
        batch_size=c.training_params.batch_size,
        shuffle=False,
        num_workers=c.training_params.num_workers,
        index_column=c.settings.index_name,
        truth_table=c.data.ice_cube.meta_table,
    )

    return dataloader


def make_test_dataloader_plus_minus(c, selection=None):
    """
    https://graphnet-team.github.io/graphnet/api/graphnet.training.utils.html#graphnet.training.utils.make_dataloader
    """
    if c.settings.is_training:
        database_path = os.path.join(c.data.dir.dataset, f"train_{c.data.ice_cube.train_batch}_db.db")
    else:
        database_path = os.path.join(c.data.dir.dataset, "test_db.db")

    db = database_path
    # selection = selection # Entire database
    pulsemaps = c.data.ice_cube.pulse_table
    features = FEATURES.KAGGLE
    truth = TRUTH.KAGGLE
    labels = None  # labels={"direction": Direction()}
    batch_size = c.training_params.batch_size
    shuffle = False
    num_workers = c.training_params.num_workers
    index_column = c.settings.index_name
    truth_table = c.data.ice_cube.meta_table

    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]

    dataset = SQLiteDataset(
        path=db,
        pulsemaps=pulsemaps,
        features=features,
        truth=truth,
        selection=selection,
        node_truth=None,
        truth_table=truth_table,
        node_truth_table=None,
        string_selection=None,
        loss_weight_table=None,
        loss_weight_column=None,
        index_column=index_column,
    )

    # adds custom labels to dataset
    if isinstance(labels, dict):
        for label in labels.keys():
            dataset.add_label(key=label, fn=labels[label])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_plus_minus,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return dataloader


def collate_fn_plus_minus(graphs: List[Data]) -> Batch:
    """Remove graphs with less than two DOM hits.

    Should not occur in "production.
    """
    batch = []
    for data in graphs:
        data.x = torch.mul(data.x, torch.FloatTensor([-1, -1, -1, 1, 1, 1]))
        data.y = -data.y
        data.z = -data.z

        if data.n_pulses > 1:
            batch.append(data)

    return Batch.from_data_list(batch)
