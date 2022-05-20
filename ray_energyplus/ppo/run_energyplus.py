#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import sys

import gym
from ray import tune
from ray.tune import register_env, Experiment

# from baselines_energyplus.bench import Monitor
from baselines_energyplus.common.energyplus_util import (
    energyplus_arg_parser,
    energyplus_logbase_dir
)
import os
import datetime

from gym_energyplus.envs import EnergyPlusEnv


def env_creator(seed):
    """
    Create gym.Env for EnergyEnv
    """
    return lambda cfg: EnergyPlusEnv(
        energyplus_file=cfg["energyplus_file"],
        model_file=cfg["model_file"],
        weather_file=cfg["weather_file"],
        log_dir=cfg["log_dir"],
        seed=seed
    )


def train(env_id, num_timesteps, seed):
    experiment_id = datetime.datetime.now().strftime("ray-%Y-%m-%d-%H-%M-%S-%f")

    log_base = energyplus_logbase_dir()
    log_dir = os.path.join(log_base, experiment_id)
    if not os.path.exists(log_dir + '/output'):
        os.makedirs(log_dir + '/output')
    os.environ["ENERGYPLUS_LOG"] = log_dir

    model = os.getenv('ENERGYPLUS_MODEL')
    if model is None:
        print('Environment variable ENERGYPLUS_MODEL is not defined')
        sys.exit(1)

    weather = os.getenv('ENERGYPLUS_WEATHER')
    if weather is None:
        print('Environment variable ENERGYPLUS_WEATHER is not defined')
        sys.exit(1)

    energyplus_file = os.getenv('ENERGYPLUS')

    env = env_creator(seed)
    register_env(env_id, env)

    config = {
        "run": "PPO",
        "env": env_id,
        "config": {
            "framework": "tf2",
            # see Ray PPO config for all options
            "gamma": 0.99,
            "kl_coeff": 0.2,
            # arguments passed to EnergyPlusEnv
            "env_config": {
                "energyplus_file": energyplus_file,
                "model_file": model,
                "weather_file": weather,
                "log_dir": log_dir
            },
        },
        "checkpoint_freq": 2,
        "checkpoint_at_end": True,
        "local_dir": log_base,
    }

    stop = {
        "timesteps_total": num_timesteps
    }

    experiment = Experiment.from_json(experiment_id, config)
    results = tune.run(run_or_experiment=experiment, stop=stop, config=config, verbose=2)
    print(results)


def main():
    args = energyplus_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
