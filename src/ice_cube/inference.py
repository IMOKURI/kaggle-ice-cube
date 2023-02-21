import logging

import pandas as pd
from .data_loader import make_test_dataloader
from .submission import to_submission_df

log = logging.getLogger(__name__)


def inference(c, model) -> pd.DataFrame:
    test_loader = make_test_dataloader(c)

    results = model.predict_as_dataframe(
        gpus=[0],
        dataloader=test_loader,
        prediction_columns=model.prediction_columns,
        additional_attributes=[c.settings.index_name],
    )

    df = to_submission_df(results)

    return df
