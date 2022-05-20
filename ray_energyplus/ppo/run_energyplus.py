#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import sys

import gym
from ray import tune
from ray.tune import register_env

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
    Create a wrapped, monitored gym.Env for EnergyEnv
    """
    # env = gym.make(env_id)
    # TODO: cannot pickle files that are not open for reading
    #       here Monitor opens a file with 'wt' mode.
    # env = Monitor(env, log_dir)
    # env.seed(seed)
    return lambda cfg: EnergyPlusEnv(
        energyplus_file=cfg["energyplus_file"],
        model_file=cfg["model_file"],
        weather_file=cfg["weather_file"],
        log_dir=cfg["log_dir"],
        seed=seed
    )


def train(env_id, num_timesteps, seed):
    log_dir = os.path.join(
        energyplus_logbase_dir(),
        datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f")
    )
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
        "env": env_id,
        "env_config": {
            "energyplus_file": energyplus_file,
            "model_file": model,
            "weather_file": weather,
            "log_dir": log_dir
        },
        "gamma": 0.5,
        "framework": "tf2",
        "kl_coeff": 0.2
    }

    stop = {
        "timesteps_total": num_timesteps
    }

    results = tune.run("PPO", stop=stop, config=config, verbose=2)
    print(results)


def main():
    args = energyplus_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
