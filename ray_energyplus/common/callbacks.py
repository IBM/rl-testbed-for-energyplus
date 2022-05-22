import json
import time
from typing import Optional, Dict

from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.agents import DefaultCallbacks
import os.path as osp
import csv

from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID


class MonitorCallbacks(DefaultCallbacks):
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

    def on_episode_start(self,
                         *,
                         worker: RolloutWorker,
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode,
                         env_index: Optional[int] = None,
                         **kwargs) -> None:
        self.time_start = time.time()

    def on_episode_end(self,
                       *,
                       worker: RolloutWorker,
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        epinfo = {
            "r": round(episode.total_reward, 6),
            "l": episode.length,
            "t": round(time.time() - self.time_start, 6)
        }
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()
