from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite import SQLiteDataset
from graphnet.training.labels import Direction
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

ICECUBE_FEATURES = FEATURES.KAGGLE + ["sensor_ratio"]


class IceCubeBatchDataset(Dataset):
    """1つのbatchをtorchのDatasetとする"""

    def __init__(
        self,
        c,
        meta_df: pd.DataFrame,
        batch_df: pd.DataFrame,
        event_ids: Optional[Union[List, pd.Index]] = None,
    ):
        super().__init__()
        self.c = c
        self.meta_df = meta_df
        self.batch_df = batch_df

        if event_ids is None:
            self.event_ids = self.batch_df["event_id"].unique()
        else:
            self.event_ids = list(set(self.batch_df["event_id"].unique()) & set(event_ids))

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):
        event_id = self.event_ids[idx]
        first_index = self.meta_df[self.meta_df["event_id"] == event_id]["first_pulse_index"].to_numpy()[0]
        last_index = self.meta_df[self.meta_df["event_id"] == event_id]["last_pulse_index"].to_numpy()[0]
        event = self.batch_df.iloc[first_index:last_index, :]
        assert len(pd.unique(event["event_id"])) == 1

        features = FEATURES.KAGGLE

        x = event[features].to_numpy()
        x = torch.tensor(x, dtype=torch.float32)

        if self.c.settings.is_training:
            azimuth = self.meta_df[self.meta_df["event_id"] == event_id]["azimuth"].to_numpy()[0]
            zenith = self.meta_df[self.meta_df["event_id"] == event_id]["zenith"].to_numpy()[0]

            data = Data(
                x=x,
                n_pulses=torch.tensor(x.shape[0], dtype=torch.int32),
                features=features,
                azimuth=torch.tensor(azimuth, dtype=torch.float32),
                zenith=torch.tensor(zenith, dtype=torch.float32),
            )
        else:
            data = Data(
                x=x,
                n_pulses=torch.tensor(x.shape[0], dtype=torch.int32),
                features=features,
            )
        return data


def make_dataloader(
    db: str,
    pulsemaps: Union[str, List[str]],
    features: List[str],
    truth: List[str],
    *,
    batch_size: int,
    shuffle: bool,
    selection: Optional[List[int]] = None,
    num_workers: int = 10,
    persistent_workers: bool = True,
    node_truth: List[str] = None,
    truth_table: str = "truth",
    node_truth_table: Optional[str] = None,
    string_selection: List[int] = None,
    loss_weight_table: Optional[str] = None,
    loss_weight_column: Optional[str] = None,
    index_column: str = "event_no",
    labels: Optional[Dict[str, Callable]] = None,
    collate_fn: Callable,
) -> DataLoader:
    """Construct `DataLoader` instance."""
    # Check(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]

    dataset = SQLiteDataset(
        path=db,
        pulsemaps=pulsemaps,
        features=features,
        truth=truth,
        selection=selection,
        node_truth=node_truth,
        truth_table=truth_table,
        node_truth_table=node_truth_table,
        string_selection=string_selection,
        loss_weight_table=loss_weight_table,
        loss_weight_column=loss_weight_column,
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
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=2,
    )

    return dataloader


def make_sqlite_dataloader(c, database_path, selection=None):
    db = database_path
    selection = selection  # Entire database
    pulsemaps = c.data.ice_cube.pulse_table
    features = FEATURES.KAGGLE
    truth = TRUTH.KAGGLE
    labels = {"direction": Direction()}
    batch_size = c.training_params.batch_size
    num_workers = c.training_params.num_workers
    index_column = c.settings.index_name
    truth_table = c.data.ice_cube.meta_table
    seed = c.global_params.seed
    test_size = 0.2

    # Reproducibility
    rng = np.random.RandomState(seed=seed)

    # Checks(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]

    if selection is None:
        # If no selection is provided, use all events in dataset.
        dataset: Dataset
        if db.endswith(".db"):
            dataset = SQLiteDataset(
                db,
                pulsemaps,
                features,
                truth,
                truth_table=truth_table,
                index_column=index_column,
            )
        else:
            raise RuntimeError(f"File {db} with format {db.split('.'[-1])} not supported.")
        selection = dataset._get_all_indices()

    # Perform train/validation split
    if isinstance(db, list):
        df_for_shuffle = pd.DataFrame({"event_no": selection, "db": None})
        shuffled_df = df_for_shuffle.sample(frac=1, replace=False, random_state=rng)
        training_df, validation_df = train_test_split(shuffled_df, test_size=test_size, random_state=rng)
        training_selection = training_df.values.tolist()
        validation_selection = validation_df.values.tolist()
    else:
        training_selection, validation_selection = train_test_split(selection, test_size=test_size, random_state=rng)

    # Create DataLoaders
    common_kwargs = dict(
        db=db,
        pulsemaps=pulsemaps,
        features=features,
        truth=truth,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=None,
        node_truth=None,
        truth_table=truth_table,
        node_truth_table=None,
        string_selection=None,
        loss_weight_column=None,
        loss_weight_table=None,
        index_column=index_column,
        labels=labels,
        collate_fn=collate_fn_training,
    )

    training_dataloader = make_dataloader(
        shuffle=True,
        selection=training_selection,
        **common_kwargs,  # type: ignore[arg-type]
    )

    validation_dataloader = make_dataloader(
        shuffle=False,
        selection=validation_selection,
        **common_kwargs,  # type: ignore[arg-type]
    )

    return (
        training_dataloader,
        validation_dataloader,
    )


def make_dataloader_batch(
    c,
    meta_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    collate_fn: Callable,
    event_ids: Optional[Union[List, pd.Index]] = None,
    is_training: bool = False,
):
    dataset = IceCubeBatchDataset(c, meta_df, batch_df, event_ids)

    dataloader = DataLoader(
        dataset,
        batch_size=c.training_params.batch_size,
        shuffle=is_training,
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


def collate_fn(graphs: List[Data]):
    # graphs = [downsample_pulse(g) for g in graphs if g.n_pulses > 1]
    graphs = [g for g in graphs if g.n_pulses > 1]
    return Batch.from_data_list(graphs)


def collate_fn_training(graphs: List[Data]):
    batch = []
    for data in graphs:
        data = downsample_pulse(data)
        data["direction"] = Direction()(data)

        if data.n_pulses > 1:
            batch.append(data)

    return Batch.from_data_list(batch)


def collate_fn_minus_minus(graphs: List[Data]):
    batch = []
    for data in graphs:
        # data = downsample_pulse(data)
        data.x = torch.mul(data.x, torch.FloatTensor([-1, -1, 1, 1, 1, 1]))

        if data.n_pulses > 1:
            batch.append(data)

    return Batch.from_data_list(batch)


def collate_fn_plus_minus(graphs: List[Data]):
    batch = []
    for data in graphs:
        # data = downsample_pulse(data)
        data.x = torch.mul(data.x, torch.FloatTensor([1, -1, 1, 1, 1, 1]))

        if data.n_pulses > 1:
            batch.append(data)

    return Batch.from_data_list(batch)


def collate_fn_minus_plus(graphs: List[Data]):
    batch = []
    for data in graphs:
        # data = downsample_pulse(data)
        data.x = torch.mul(data.x, torch.FloatTensor([-1, 1, 1, 1, 1, 1]))

        if data.n_pulses > 1:
            batch.append(data)

    return Batch.from_data_list(batch)
