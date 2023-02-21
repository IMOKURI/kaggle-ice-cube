import logging

import hydra

import src.utils as utils
from src.ice_cube.data_loader import make_test_dataloader
from src.ice_cube.model import load_pretrained_model
from src.ice_cube.submission import to_submission_df

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    if c.settings.in_kaggle:
        c.data.dir.pretrained = "/kaggle/input/dynedge-pretrained"

    utils.basic_environment_info()
    utils.fix_seed(utils.choice_seed(c))

    test_loader = make_test_dataloader(c)
    model = load_pretrained_model(c)
    results = model.predict_as_dataframe(
        gpus=[0],
        dataloader=test_loader,
        prediction_columns=model.prediction_columns,
        additional_attributes=[c.settings.index_name],
    )

    submission_df = to_submission_df(results)
    submission_df.to_csv("submission.csv")

    log.info("Done.")


if __name__ == "__main__":
    main()
