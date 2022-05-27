"""
Helpers for script run_energyplus.py.
"""
import os
import glob
# following import necessary to register EnergyPlus-v0 env
import gym_energyplus  # noqa


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def energyplus_arg_parser():
    """
    Create an argparse.ArgumentParser for run_energyplus.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', '-e', help='environment ID', type=str, default='EnergyPlus-v0')
    parser.add_argument('--seed', '-s', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(100))
    parser.add_argument('--save-interval', type=int, default=int(0))
    parser.add_argument('--model-pickle', help='model pickle', type=str, default='')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='')
    return parser


def energyplus_locate_log_dir(index=0):
    pat_openai = energyplus_logbase_dir() + f'/openai-????-??-??-??-??-??-??????*/progress.csv'
    pat_ray = energyplus_logbase_dir() + f'/ray-????-??-??-??-??-??-??????*/*/progress.csv'
    files = [
        (f, os.path.getmtime(f))
        for pat in [pat_openai, pat_ray]
        for f in glob.glob(pat)
    ]
    newest = sorted(files, key=lambda files: files[1])[-(1 + index)][0]
    dir = os.path.dirname(newest)
    # in ray, progress.csv is in a subdir, so we need to get
    # one step upper.
    if "/ray-" in dir:
        dir = os.path.dirname(dir)
    print('energyplus_locate_log_dir: {}'.format(dir))
    return dir


def energyplus_logbase_dir():
    logbase_dir = os.getenv('ENERGYPLUS_LOGBASE')
    if logbase_dir is None:
        logbase_dir = '/tmp'
    return logbase_dir
