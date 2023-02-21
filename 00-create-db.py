import logging

import hydra

import src.utils as utils
from src.ice_cube.sqlite import Sqlite

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    utils.basic_environment_info()
    utils.fix_seed(utils.choice_seed(c))

    s = Sqlite(c)
    s.convert_to_sqlite()

    log.info("Done.")


if __name__ == "__main__":
    main()
