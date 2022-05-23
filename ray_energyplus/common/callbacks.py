import json
import time
from typing import Dict, Any

import os.path as osp
import csv

from ray.tune.logger import LoggerCallback
from ray.tune.trial import Trial


class MonitorCallbacks(LoggerCallback):
    EXT = "monitor.csv"
    f = None

    def __init__(self, filename, env_id):
        super().__init__()

        self.time_start = time.time()

        print('Monitor: filename={}'.format(filename))
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(MonitorCallbacks.EXT):
                if osp.isdir(filename):
                    filename = osp.join(filename, MonitorCallbacks.EXT)
                else:
                    filename = filename + "." + MonitorCallbacks.EXT
            self.f = open(filename, "wt")
            self.f.write('#%s\n' % json.dumps({"t_start": self.time_start, 'env_id': env_id}))
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't'))
            self.logger.writeheader()
            self.f.flush()

    def log_trial_start(self, trial: Trial):
        self.time_start = time.time()

    def log_trial_result(self, iteration: int, trial: Trial, result: Dict[str, Any]):
        # print("iteration", iteration)
        # print("result", result)
        t = time.time()

        epinfo = {
            "r": round(result["episode_reward_mean"], 6),
            "l": result["episode_len_mean"],
            "t": round(t - self.time_start, 6)
        }
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()

        self.time_start = t

    def log_trial_end(self, trial: Trial, failed: bool = False):
        if self.f:
            self.f.flush()
            self.f.close()
