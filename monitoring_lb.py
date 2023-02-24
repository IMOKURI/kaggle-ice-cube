import datetime
import logging
import time
from datetime import timezone

from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(
    # filename=__file__.replace('.py', '.log'),
    level=logging.getLevelName("INFO"),
    format="%(asctime)s [%(levelname)s] [%(module)s] %(message)s",
)

log = logging.getLogger(__name__)

api = KaggleApi()
api.authenticate()

COMPETITION = "icecube-neutrinos-in-deep-ice"
result_ = api.competition_submissions(COMPETITION)[0]
latest_ref = str(result_)  # 最新のサブミット番号
submit_time = result_.date

status = ""

while status != "complete":
    list_of_submission = api.competition_submissions(COMPETITION)
    for result in list_of_submission:
        if str(result.ref) == latest_ref:
            break
    status = result.status

    now = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
    elapsed_time = int((now - submit_time).seconds / 60) + 1
    if status == "complete":
        log.info(f"run-time: {elapsed_time} min, LB: {result.publicScore}")
    else:
        log.info(f"elapsed time: {elapsed_time} min")
        time.sleep(60)
