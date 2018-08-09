#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import argparse

from baselines_energyplus.common.energyplus_util import make_energyplus_env, energyplus_locate_log_dir
#import os
import gym_energyplus

def plot_energyplus_arg_parser():
    """
    Create an argparse.ArgumentParser for plot_energyplus.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', '-e', help='environment ID', type=str, default='EnergyPlus-v0')
    parser.add_argument('--log-dir', '-l', help='Plot all data in the log directory', type=str, default='')
    parser.add_argument('--csv-file', '-c', help='Plot single CSV file', type=str, default='')
    parser.add_argument('--dump-timesteps', '-d', help='Dump timesteps', action='store_true')
    parser.add_argument('--dump-episodes', '-D', help='Dump episodes', action='store_true')
    return parser

def energyplus_plot(env_id, log_dir='', csv_file='', dump_timesteps=False, dump_episodes=False):
    if log_dir is not '' and csv_file is not '':
        print('Either log directory or csv file can be specified')
        os.exit(1)
    if log_dir is '' and csv_file is '':
        log_dir = energyplus_locate_log_dir()
    env = make_energyplus_env(env_id, 0)
    assert sum([dump_timesteps, dump_episodes]) != 2
    if dump_timesteps:
        env.env.dump_timesteps(log_dir=log_dir, csv_file=csv_file)
    elif dump_episodes:
        env.env.dump_episodes(log_dir=log_dir, csv_file=csv_file)
    else:
        env.env.plot(log_dir=log_dir, csv_file=csv_file)
    env.close()

def main():
    args = plot_energyplus_arg_parser().parse_args()
    energyplus_plot(args.env, log_dir=args.log_dir, csv_file=args.csv_file, dump_timesteps=args.dump_timesteps, dump_episodes=args.dump_episodes)

if __name__ == '__main__':
    main()
