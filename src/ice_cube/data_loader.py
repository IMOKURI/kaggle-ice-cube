from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.training.labels import Direction
from graphnet.training.utils import make_dataloader, make_train_validation_dataloader
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data


class IceCubeBatchDataset(Dataset):
    """1つのbatchをtorchのDatasetとする"""

    def __init__(
        self, c, batch_id: int, meta_df: pd.DataFrame, sensor_df: pd.DataFrame, event_ids: Optional[List] = None
    ):
        super().__init__()
        self.c = c
        self.batch_id = batch_id
        self.meta_df = meta_df
        self.sensor_df = sensor_df

        if c.settings.is_training:
            self.input_batch_dir = c.data.dir.input_train
        else:
            self.input_batch_dir = c.data.dir.input_test
        self.batch_df = pd.read_parquet(path=f"{self.input_batch_dir}/batch_{batch_id}.parquet").reset_index()

        if event_ids is None:
            self.event_ids = self.batch_df["event_id"].unique()
        else:
            self.event_ids = list(set(self.batch_df["event_id"].unique()) & set(event_ids))

        self._preprocess()

    def _preprocess(self):
        self.batch_df = pd.merge(self.batch_df, self.sensor_df, on="sensor_id").sort_values("event_id")
        self.batch_df["auxiliary"] = self.batch_df["auxiliary"].replace({True: 1, False: 0})

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):
        event_id = self.event_ids[idx]
        first_index = self.meta_df[self.meta_df["event_id"] == event_id]["first_pulse_index"].to_numpy()[0]
        last_index = self.meta_df[self.meta_df["event_id"] == event_id]["last_pulse_index"].to_numpy()[0]
        event = self.batch_df.iloc[first_index:last_index, :]
        # assert len(pd.unique(event["event_id"])) == 1

        x = event[FEATURES.KAGGLE].to_numpy()
        x = torch.tensor(x, dtype=torch.float32)

        if self.c.settings.is_training:
            azimuth = self.meta_df[self.meta_df["event_id"] == event_id]["azimuth"].to_numpy()[0]
            zenith = self.meta_df[self.meta_df["event_id"] == event_id]["zenith"].to_numpy()[0]

            data = Data(
                x=x,
                n_pulses=torch.tensor(x.shape[0], dtype=torch.int32),
                features=FEATURES.KAGGLE,
                azimuth=torch.tensor(azimuth, dtype=torch.float32),
                zenith=torch.tensor(zenith, dtype=torch.float32),
            )
        else:
            data = Data(
                x=x,
                n_pulses=torch.tensor(x.shape[0], dtype=torch.int32),
                features=FEATURES.KAGGLE,
            )
        return data


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
        test_size=0.2,
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


def make_test_dataloader_batch(
    c,
    batch_id: int,
    meta_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    collate_fn: Callable,
    event_ids: Optional[List] = None,
):
    dataset = IceCubeBatchDataset(c, batch_id, meta_df, sensor_df, event_ids)

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


def downsample_pulse(data: Data) -> Data:
    pulse_limit = 300
    if data.n_pulses > pulse_limit:
        data.x = data.x[np.random.choice(data.n_pulses, pulse_limit)]
        data.n_pulses = torch.tensor(pulse_limit, dtype=torch.int32)
    return data


def collate_fn(graphs: List[Data]) -> Batch:
    graphs = [downsample_pulse(g) for g in graphs if g.n_pulses > 1]
    return Batch.from_data_list(graphs)


def collate_fn_training(graphs: List[Data]) -> Batch:
    batch = []
    for data in graphs:
        data = downsample_pulse(data)
        data["direction"] = Direction()(data)

        if data.n_pulses > 1:
            batch.append(data)

    return Batch.from_data_list(batch)


def collate_fn_minus_minus(graphs: List[Data]) -> Batch:
    batch = []
    for data in graphs:
        data = downsample_pulse(data)
        data.x = torch.mul(data.x, torch.FloatTensor([-1, -1, 1, 1, 1, 1]))

        if data.n_pulses > 1:
            batch.append(data)

    return Batch.from_data_list(batch)


def collate_fn_plus_minus(graphs: List[Data]) -> Batch:
    batch = []
    for data in graphs:
        data = downsample_pulse(data)
        data.x = torch.mul(data.x, torch.FloatTensor([1, -1, 1, 1, 1, 1]))

        if data.n_pulses > 1:
            batch.append(data)

    return Batch.from_data_list(batch)


def collate_fn_minus_plus(graphs: List[Data]) -> Batch:
    batch = []
    for data in graphs:
        data = downsample_pulse(data)
        data.x = torch.mul(data.x, torch.FloatTensor([-1, 1, 1, 1, 1, 1]))

        if data.n_pulses > 1:
            batch.append(data)

    return Batch.from_data_list(batch)
