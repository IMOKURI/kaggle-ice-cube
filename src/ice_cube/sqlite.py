# https://www.kaggle.com/code/rasmusrse/graphnet-baseline-submission

import logging
import os
import sqlite3
from typing import List

import pandas as pd
import pyarrow.parquet as pq
import sqlalchemy
from graphnet.data.sqlite.sqlite_utilities import create_table
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import euclidean
from tqdm import tqdm

log = logging.getLogger(__name__)


class Sqlite:
    def __init__(self, c, stage2=False):
        self.enable_h_cluster = False  # c.preprocess_params.enable_h_cluster
        self.stage2 = stage2

        if c.settings.is_training:
            self.input_batch_dir = c.data.dir.input_train
            self.metadata_path = os.path.join(c.data.dir.input, "train_meta.parquet")
            self.database_path = os.path.join(c.data.dir.dataset, f"train_{c.data.ice_cube.train_batch}_db.db")
        else:
            self.input_batch_dir = c.data.dir.input_test
            self.metadata_path = os.path.join(c.data.dir.input, "test_meta.parquet")
            self.database_path = os.path.join(c.data.dir.dataset, "test_db.db")

        if stage2:
            self.database_path = self.database_path.replace("_db.", "_db2.")
            self.enable_h_cluster = c.preprocess_params.enable_h_cluster
            high_sigma = pd.read_csv("submission_high_sigma.csv")
            self.high_sigma_id = high_sigma["event_id"].astype(int)

        self.geometry_table = pd.read_csv(os.path.join(c.data.dir.input, "sensor_geometry.csv"))

        self.meta_table = c.data.ice_cube.meta_table
        self.pulse_table = c.data.ice_cube.pulse_table

        self.is_training = c.settings.is_training
        self.train_batch = list(range(c.data.ice_cube.train_batch, c.data.ice_cube.train_batch + 1))

        os.makedirs(c.data.dir.dataset, exist_ok=True)

    def convert_to_sqlite(self, batch_size: int = 200000):
        metadata_iter = pq.ParquetFile(self.metadata_path).iter_batches(batch_size=batch_size)

        for n, meta_df in enumerate(metadata_iter):
            if self.is_training and n not in self.train_batch:
                continue

            meta_df = meta_df.to_pandas()
            if self.stage2:
                meta_df = meta_df[meta_df["event_id"].isin(self.high_sigma_id)]

            self.create_table(columns=meta_df.columns, table_name=self.meta_table, is_primary_key=True)
            self.add_records(df=meta_df, table_name=self.meta_table)

            batch_id = pd.unique(meta_df["batch_id"])
            assert len(batch_id) == 1, "contains multiple batch_ids. Did you set the batch_size correctly?"
            log.info(f"{self.database_path} . {batch_id} ...")

            batch_df = pd.read_parquet(path=f"{self.input_batch_dir}/batch_{batch_id[0]}.parquet").reset_index()
            if self.stage2:
                batch_df = batch_df[batch_df["event_id"].isin(self.high_sigma_id)]

            sensor_positions = self.geometry_table.loc[batch_df["sensor_id"], ["x", "y", "z"]]
            sensor_positions.index = batch_df.index

            for column in sensor_positions.columns:
                if column not in batch_df.columns:
                    batch_df[column] = sensor_positions[column]

            batch_df["auxiliary"] = batch_df["auxiliary"].replace({True: 1, False: 0})

            self.create_table(columns=batch_df.columns, table_name=self.pulse_table, is_primary_key=False)
            self.load_events(meta_df, batch_df)

            del meta_df
            del batch_df
        del metadata_iter
        log.info(f"Conversion Complete!. Database available at {self.database_path}")

    def create_table(
        self,
        columns: List,
        table_name: str,
        is_primary_key: bool,
    ) -> None:
        try:
            create_table(
                columns=columns,
                database_path=self.database_path,
                table_name=table_name,
                integer_primary_key=is_primary_key,
                index_column="event_id",
            )
        except sqlite3.OperationalError as e:
            if "already exists" in str(e):
                pass
            else:
                raise e

    def add_records(
        self,
        df: pd.DataFrame,
        table_name: str,
    ) -> None:
        engine = sqlalchemy.create_engine("sqlite:///" + self.database_path)
        df.to_sql(table_name, con=engine, index=False, if_exists="append", chunksize=200000)
        engine.dispose()
        return

    def load_events(self, meta_df: pd.DataFrame, batch_df: pd.DataFrame):
        for event_id in tqdm(batch_df["event_id"].unique()):
            if self.stage2:
                event_df = batch_df[batch_df["event_id"] == event_id].copy()
            else:
                first_index = meta_df[meta_df["event_id"] == event_id]["first_pulse_index"].to_numpy()[0]
                last_index = meta_df[meta_df["event_id"] == event_id]["last_pulse_index"].to_numpy()[0]
                event_df = batch_df.iloc[first_index:last_index, :].copy()

            if self.enable_h_cluster:
                try:
                    h_cluster = linkage(event_df[["time", "x", "y", "z"]], metric=world_dist)
                    event_df.loc[:, "h_label"] = fcluster(h_cluster, 1)

                    # event_df = event_df[event_df.duplicated(subset=["h_label"], keep=False)]
                    event_df = event_df[event_df["h_label"] == event_df["h_label"].value_counts().idxmax()]
                    event_df.drop(["h_label"], axis=1, inplace=True)
                except Exception as e:
                    log.error(f"event_id: {event_id}, error: {e}")
                    event_df = batch_df.iloc[first_index:last_index, :].copy()

            assert len(event_df) > 0
            self.add_records(event_df, self.pulse_table)


def world_dist(u, v):
    """
    2つのベクトルの世界距離を計算する
    各ベクトルは時刻(1次元)と空間(3次元)の値を持つこと。
    """
    assert u.shape[0] == 4
    assert v.shape[0] == 4

    # ここでは時間は ナノ秒単位
    # 光は 1ナノ秒で 約0.3m 進む
    time = 0.3 * (u[0] - v[0]) ** 2 + 1e-8
    space = euclidean(u[1:], v[1:])
    return space / time
