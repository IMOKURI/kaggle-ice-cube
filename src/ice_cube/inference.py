import logging
from typing import Any, Dict

import pandas as pd
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.training.utils import make_dataloader

log = logging.getLogger(__name__)


def inference(model, config: Dict[str, Any]) -> pd.DataFrame:
    """Applies model to the database specified in config['inference_database_path'] and saves results to disk."""
    # Make Dataloader
    test_dataloader = make_dataloader(
        db=config["inference_database_path"],
        selection=None,  # Entire database
        pulsemaps=config["pulsemap"],
        features=FEATURES.KAGGLE,
        truth=TRUTH.KAGGLE,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        labels=None,  # Cannot make labels in test data
        index_column=config["index_column"],
        truth_table=config["truth_table"],
    )

    # Get predictions
    results = model.predict_as_dataframe(
        gpus=[0],
        dataloader=test_dataloader,
        prediction_columns=model.prediction_columns,
        additional_attributes=["event_id"],
    )
    return results
