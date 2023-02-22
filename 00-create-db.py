import logging
import os

import hydra

import src.utils as utils
from src.ice_cube.sqlite import Sqlite

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    if c.settings.in_kaggle:
        c.settings.is_training = False

    utils.basic_environment_info()
    utils.fix_seed(utils.choice_seed(c))

    s = Sqlite(c)
    if not os.path.exists(s.database_path):
        s.convert_to_sqlite()

    log.info("Done.")


if __name__ == "__main__":
    main()
