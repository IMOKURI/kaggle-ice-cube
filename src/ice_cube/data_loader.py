from typing import Callable, List

import torch
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite import SQLiteDataset
from graphnet.training.labels import Direction
from graphnet.training.utils import make_dataloader, make_train_validation_dataloader
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data


def make_train_dataloader(c, database_path, selection=None):
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


def make_test_dataloader(c, database_path, selection=None):
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


def make_test_dataloader_custom(c, database_path, collate_fn: Callable, selection=None):
    dataset = SQLiteDataset(
        path=database_path,
        pulsemaps=[c.data.ice_cube.pulse_table],
        features=FEATURES.KAGGLE,
        truth=TRUTH.KAGGLE,
        selection=selection,
        node_truth=None,
        truth_table=c.data.ice_cube.meta_table,
        node_truth_table=None,
        string_selection=None,
        loss_weight_table=None,
        loss_weight_column=None,
        index_column=c.settings.index_name,
    )

    # adds custom labels to dataset
    labels = None  # labels={"direction": Direction()}
    if isinstance(labels, dict):
        for label in labels.keys():
            dataset.add_label(key=label, fn=labels[label])

    dataloader = DataLoader(
        dataset,
        batch_size=c.training_params.batch_size,
        shuffle=False,
        num_workers=c.training_params.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return dataloader


def collate_fn(graphs: List[Data]) -> Batch:
    graphs = [g for g in graphs if g.n_pulses > 1]
    return Batch.from_data_list(graphs)


def collate_fn_plus_minus(graphs: List[Data]) -> Batch:
    batch = []
    for data in graphs:
        data.x = torch.mul(data.x, torch.FloatTensor([-1, -1, -1, 1, 1, 1]))
        data.y = -data.y
        data.z = -data.z

        if data.n_pulses > 1:
            batch.append(data)

    return Batch.from_data_list(batch)
