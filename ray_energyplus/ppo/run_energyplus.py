#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import sys

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
from ray_energyplus.common.callbacks import MonitorCallbacks


def env_creator(seed):
    """
    Create gym.Env for EnergyEnv
    """

    def create_env(cfg):
        env_cfg = {
            k: cfg[k] for k in [
                "energyplus_file",
                "model_file",
                "weather_file",
                "log_dir",
                "framework"
            ]
        }
        env_cfg["seed"] = seed

        return EnergyPlusEnv(**env_cfg)

    return create_env


def train(env_id, num_timesteps, seed):
    experiment_id = datetime.datetime.now().strftime("ray-%Y-%m-%d-%H-%M-%S-%f")

    log_base = energyplus_logbase_dir()
    log_dir = os.path.join(log_base, experiment_id)
    if not os.path.exists(log_dir + "/output"):
        os.makedirs(log_dir + "/output")
    os.environ["ENERGYPLUS_LOG"] = log_dir

    model = os.getenv("ENERGYPLUS_MODEL")
    if model is None:
        print("Environment variable ENERGYPLUS_MODEL is not defined")
        sys.exit(1)

    weather = os.getenv("ENERGYPLUS_WEATHER")
    if weather is None:
        print("Environment variable ENERGYPLUS_WEATHER is not defined")
        sys.exit(1)

    energyplus_file = os.getenv("ENERGYPLUS")
    if energyplus_file is None:
        print("Environment variable ENERGYPLUS is not defined")
        sys.exit(1)

    # gym env registration
    env = env_creator(seed)
    register_env(env_id, env)

    # experiment configuration, including alg params
    framework = "tf2"
    config = {
        "run": "PPO",
        "env": env_id,
        "config": {
            "framework": framework,
            # see Ray PPO config for all options
            "gamma": 0.99,
            "kl_coeff": 0.2,
            # if true, use the Generalized Advantage Estimator (GAE)
            # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            "use_gae": True,
            # GAE(lambda) parameter
            "lambda": 0.97,
            # increase this if more CPUs are available
            "num_workers": 2,
            # small batches speed up training
            # you may want to increase these values for stability
            "train_batch_size": 128,
            "rollout_fragment_length": 64,
            # magnitude of reward values require large
            # value function clipping param (otherwise would take
            # large number of iterations to converge)
            "vf_clip_param": 1000,
            # it's an (almost) continuous control problem
            # we split each year-long episode into 1-day logical episodes
            # so policy is updated more frequently.
            # this setting is optional, you may want to comment / remove
            # if you want to revert to full-episode setting
            "horizon": 96,
            # do not force env reset when horizon is hit
            "soft_horizon": True,
            # rescale observations
            "observation_filter": "MeanStdFilter",
            # arguments passed to EnergyPlusEnv
            "env_config": {
                "energyplus_file": energyplus_file,
                "model_file": model,
                "weather_file": weather,
                "log_dir": log_dir,
                "framework": framework
            },
        },
        # checkpoint every 2 iterations, and at end of experiment
        # this allows to reload the policy model
        "checkpoint_freq": 2,
        "checkpoint_at_end": True,
        # dir where tune will write its output
        "local_dir": log_base,
    }

    # experiment stop conditions
    stop = {
        "timesteps_total": num_timesteps
    }

    # callbacks
    tune_callbacks = [MonitorCallbacks(filename=log_dir, env_id=env_id)]

    experiment = Experiment.from_json(experiment_id, config)
    results = tune.run(
        run_or_experiment=experiment,
        stop=stop,
        config=config,
        callbacks=tune_callbacks,
        verbose=2
    )
    print(results.results_df)


def main():
    args = energyplus_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
