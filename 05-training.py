import logging
import os

import hydra
import pandas as pd
import pyarrow.parquet as pq
from graphnet.training.callbacks import ProgressBar
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import src.utils as utils
from src.ice_cube.data_loader import CollateFn, make_dataloader_batch, make_sqlite_dataloader
from src.ice_cube.model import build_model, load_pretrained_model
from src.preprocess import preprocess

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

    c.training_params.batch_size = 512

    if c.training_params.stage2:
        log.info("Stage2 training.")
        # results = pd.read_csv("results.csv").set_index("event_id")
        # log.info(f"Num of data: {len(results)}")

        metadata_path = os.path.join(c.data.dir.input, "train_meta.parquet")
        sensor_df = pd.read_csv(os.path.join(c.data.dir.input, "sensor_geometry.csv"))

        batch_size = 200_000
        metadata_iter = pq.ParquetFile(metadata_path).iter_batches(batch_size=batch_size)

        train_batch = list(
            range(c.data.ice_cube.train_batch, c.data.ice_cube.train_batch + c.data.ice_cube.train_batch_size)
        )

        if c.settings.is_training:
            input_batch_dir = c.data.dir.input_train
        else:
            input_batch_dir = c.data.dir.input_test

        all_meta_df = None
        all_batch_df = None

        for n, meta_df in enumerate(metadata_iter):
            if c.settings.is_training and n not in train_batch:
                continue

            meta_df = meta_df.to_pandas()

            batch_id = pd.unique(meta_df["batch_id"])
            assert len(batch_id) == 1, "contains multiple batch_ids. Did you set the batch_size correctly?"
            log.info(f"Batch {batch_id} ...")

            batch_df = pd.read_parquet(path=f"{input_batch_dir}/batch_{batch_id[0]}.parquet").reset_index()
            # ここで必要なデータだけに絞りたいがそうすると pulse_index がずれる。
            # batch_df = batch_df[batch_df["event_id"].isin(results.index)]

            # meta_df = meta_df[meta_df["event_id"].isin(results.index)]

            log.info(f"Meta size: {len(meta_df)}, Batch size: {len(batch_df)}")

            if all_meta_df is None:
                all_meta_df = meta_df
            else:
                meta_df.loc[:, ["first_pulse_index", "last_pulse_index"]] = meta_df.loc[
                    :, ["first_pulse_index", "last_pulse_index"]
                ] + len(all_batch_df)
                all_meta_df = pd.concat([all_meta_df, meta_df])

            if all_batch_df is None:
                all_batch_df = batch_df
            else:
                all_batch_df = pd.concat([all_batch_df, batch_df])

        assert all_meta_df is not None

        all_batch_df = pd.merge(all_batch_df, sensor_df, on="sensor_id").sort_values("event_id")
        all_batch_df = preprocess(c, all_batch_df, "batch")

        # train_idx, valid_idx = train_test_split(results.index, test_size=0.2)
        train_idx, valid_idx = train_test_split(all_meta_df["event_id"], test_size=0.2)

        train_loader = make_dataloader_batch(
            c, all_meta_df, all_batch_df, CollateFn(is_training=True), train_idx, is_training=True
        )
        valid_loader = make_dataloader_batch(c, all_meta_df, all_batch_df, CollateFn(is_training=True), valid_idx)
        log.info(f"Train size: {len(train_loader.dataset)}, Valid size: {len(valid_loader.dataset)}")

    else:
        database_path = os.path.join(c.data.dir.dataset, f"train_{c.data.ice_cube.train_batch}_db.db")
        train_loader, valid_loader = make_sqlite_dataloader(c, database_path)

    model = build_model(c, train_loader)
    # model = load_pretrained_model(c, train_loader, state_dict_path=c.inference_params.model_path)

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
    # model.save(os.path.join(run_dir, "model.pth"))

    log.info("Done.")


if __name__ == "__main__":
    main()
