import logging

import hydra

import src.utils as utils
from src.ice_cube.sqlite import Sqlite

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(c):
    utils.basic_environment_info()
    utils.debug_settings(c)
    run = utils.setup_wandb(c)

    utils.fix_seed(utils.choice_seed(c))

    s = Sqlite(c)
    s.convert_to_sqlite()

    log.info("Done.")

    utils.teardown_wandb(c, run, 0, 0)


if __name__ == "__main__":
    main()
