# https://www.kaggle.com/code/rasmusrse/graphnet-baseline-submission

import logging
import os
import sqlite3

import pandas as pd
import pyarrow.parquet as pq
import sqlalchemy
from graphnet.data.sqlite.sqlite_utilities import create_table

log = logging.getLogger(__name__)


class Sqlite:
    def __init__(self, c):
        if c.settings.is_training:
            self.input_batch_dir = c.data.dir.input_train
            self.metadata_path = os.path.join(c.data.dir.input, "train_meta.parquet")
            self.database_path = os.path.join(c.data.dir.working, "train_db.db")
        else:
            self.input_batch_dir = c.data.dir.input_test
            self.metadata_path = os.path.join(c.data.dir.input, "test_meta.parquet")
            self.database_path = os.path.join(c.data.dir.working, "test_db.db")

        self.geometry_table = pd.read_csv(os.path.join(c.data.dir.input, "sensor_geometry.csv"))

        self.meta_table = c.data.ice_cube.meta_table
        self.pulse_table = c.data.ice_cube.pulse_table

    def convert_to_sqlite(self, batch_size: int = 200000) -> None:
        """Converts a selection of the Competition's parquet files to a single sqlite database.

        Args:
            batch_size (int): the number of rows extracted from meta data file at a time. Keep low for memory efficiency.
        """
        metadata_iter = pq.ParquetFile(self.metadata_path).iter_batches(batch_size=batch_size)

        for metadata_batch in metadata_iter:
            metadata_batch = metadata_batch.to_pandas()
            self.add_to_table(df=metadata_batch, table_name=self.meta_table, is_primary_key=True)
            pulses = self.load_input(meta_batch=metadata_batch)
            del metadata_batch  # memory
            self.add_to_table(df=pulses, table_name=self.pulse_table, is_primary_key=False)
            del pulses  # memory
        del metadata_iter  # memory
        log.info(f"Conversion Complete!. Database available at\n {self.database_path}")

    def add_to_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        is_primary_key: bool,
    ) -> None:
        """Writes meta data to sqlite table.

        Args:
            df (pd.DataFrame): the dataframe that is being written to table.
            table_name (str, optional): The name of the meta table. Defaults to 'meta_table'.
            is_primary_key(bool): Must be True if each row of df corresponds to a unique event_id. Defaults to False.
        """
        try:
            create_table(
                columns=df.columns,
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
        engine = sqlalchemy.create_engine("sqlite:///" + self.database_path)
        df.to_sql(table_name, con=engine, index=False, if_exists="append", chunksize=200000)
        engine.dispose()
        return

    def load_input(self, meta_batch: pd.DataFrame) -> pd.DataFrame:
        """
        Will load the corresponding detector readings associated with the meta data batch.
        """
        batch_id = pd.unique(meta_batch["batch_id"])
        log.info(f"{self.database_path} . {batch_id} ...")

        assert len(batch_id) == 1, "contains multiple batch_ids. Did you set the batch_size correctly?"

        detector_readings = pd.read_parquet(path=f"{self.input_batch_dir}/batch_{batch_id[0]}.parquet")
        sensor_positions = self.geometry_table.loc[detector_readings["sensor_id"], ["x", "y", "z"]]
        sensor_positions.index = detector_readings.index

        for column in sensor_positions.columns:
            if column not in detector_readings.columns:
                detector_readings[column] = sensor_positions[column]

        detector_readings["auxiliary"] = detector_readings["auxiliary"].replace({True: 1, False: 0})
        return detector_readings.reset_index()
