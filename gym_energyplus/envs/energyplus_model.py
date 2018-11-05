# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from abc import ABCMeta, abstractmethod
import os, sys, time
from scipy.special import expit
import numpy as np
from datetime import datetime, timedelta
from glob import glob
from matplotlib.widgets import Slider, Button, RadioButtons
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
import json

class EnergyPlusModel(metaclass=ABCMeta):

    def __init__(self,
                 model_file,
                 log_dir=None,
                 verbose=False):
        self.log_dir = log_dir
        self.model_basename = os.path.splitext(os.path.basename(model_file))[0]
        self.setup_spaces()
        self.action = 0.5 * (self.action_space.low + self.action_space.high)
        self.action_prev = self.action
        self.raw_state = None
        self.verbose = verbose
        self.timestamp_csv = None
        self.sl_episode = None

        # Progress data
        self.num_episodes = 0
        self.num_episodes_last = 0

        self.reward = None
        self.reward_mean = None

    def reset(self):
        pass

    # Parse date/time format from EnergyPlus and return datetime object with correction for 24:00 case
    def _parse_datetime(self, dstr):
        # ' MM/DD  HH:MM:SS' or 'MM/DD  HH:MM:SS'
        # Dirty hack
        if dstr[0] != ' ':
            dstr = ' ' + dstr
        #year = 2017
        year = 2013 # for CHICAGO_IL_USA TMY2-94846
        month = int(dstr[1:3])
        day = int(dstr[4:6])
        hour = int(dstr[8:10])
        minute = int(dstr[11:13])
        sec = 0
        msec = 0
        if hour == 24:
            hour = 0
            dt = datetime(year, month, day, hour, minute, sec, msec) + timedelta(days=1)
        else:
            dt = datetime(year, month, day, hour, minute, sec, msec)
        return dt

    # Convert list of date/time string to list of datetime objects
    def _convert_datetime24(self, dates):
        # ' MM/DD  HH:MM:SS'
        dates_new = []
        for d in dates:
            #year = 2017
            #month = int(d[1:3])
            #day = int(d[4:6])
            #hour = int(d[8:10])
            #minute = int(d[11:13])
            #sec = 0
            #msec = 0
            #if hour == 24:
            #    hour = 0
            #    d_new = datetime(year, month, day, hour, minute, sec, msec) + dt.timedelta(days=1)
            #else:
            #    d_new = datetime(year, month, day, hour, minute, sec, msec)
            #dates_new.append(d_new)
            dates_new.append(self._parse_datetime(d))
        return dates_new

    # Generate x_pos and x_labels
    def generate_x_pos_x_labels(self, dates):
        time_delta  = self._parse_datetime(dates[1]) - self._parse_datetime(dates[0])
        x_pos = []
        x_labels = []
        for i, d in enumerate(dates):
            dt = self._parse_datetime(d) - time_delta
            if dt.hour == 0 and dt.minute == 0:
                x_pos.append(i)
                x_labels.append(dt.strftime('%m/%d'))
        return x_pos, x_labels

    def set_action(self, normalized_action):
        # In TPRO/POP1/POP2 in baseline, action seems to be normalized to [-1.0, 1.0].
        # So it must be scaled back into action_space by the environment.
        self.action_prev = self.action
        self.action = self.action_space.low + (normalized_action + 1.) * 0.5 * (self.action_space.high - self.action_space.low)
        self.action = np.clip(self.action, self.action_space.low, self.action_space.high)

    @abstractmethod
    def setup_spaces(self): pass

    # Need to handle the case that raw_state is None
    @abstractmethod
    def set_raw_state(self, raw_state): pass

    def get_state(self):
        return self.format_state(self.raw_state)

    @abstractmethod
    def compute_reward(self): pass

    @abstractmethod
    def format_state(self, raw_state): pass

    #--------------------------------------------------
    # Plotting staffs follow
    #--------------------------------------------------
    def plot(self, log_dir='', csv_file='', **kwargs):
        if log_dir is not '':
            if not os.path.isdir(log_dir):
                print('energyplus_model.plot: {} is not a directory'.format(log_dir))
                return
            print('energyplus_plot.plot log={}'.format(log_dir))
            self.log_dir = log_dir
            self.show_progress()
        else:
            if not os.path.isfile(csv_file):
                print('energyplus_model.plot: {} is not a file'.format(csv_file))
                return
            print('energyplus_model.plot csv={}'.format(csv_file))
            self.read_episode(csv_file)
            plt.rcdefaults()
            plt.rcParams['font.size'] = 6
            plt.rcParams['lines.linewidth'] = 1.0
            plt.rcParams['legend.loc'] = 'lower right'
            self.fig = plt.figure(1, figsize=(16, 10))
            self.plot_episode(csv_file)
            plt.show()

    # Show convergence
    def show_progress(self):
        self.monitor_file = self.log_dir + '/monitor.csv'

        # Read progress file
        if not self.read_monitor_file():
            print('Progress data is missing')
            sys.exit(1)

        # Initialize graph
        plt.rcdefaults()
        plt.rcParams['font.size'] = 6
        plt.rcParams['lines.linewidth'] = 1.0
        plt.rcParams['legend.loc'] = 'lower right'

        self.fig = plt.figure(1, figsize=(16, 10))

        # Show widgets
        axcolor = 'lightgoldenrodyellow'
        self.axprogress = self.fig.add_axes([0.15, 0.10, 0.70, 0.15], facecolor=axcolor)
        self.axslider = self.fig.add_axes([0.15, 0.04, 0.70, 0.02], facecolor=axcolor)
        axfirst = self.fig.add_axes([0.15, 0.01, 0.03, 0.02])
        axlast = self.fig.add_axes([0.82, 0.01, 0.03, 0.02])
        axprev = self.fig.add_axes([0.46, 0.01, 0.03, 0.02])
        axnext = self.fig.add_axes([0.51, 0.01, 0.03, 0.02])

        # Slider is drawn in plot_progress()

        # First/Last button
        self.button_first = Button(axfirst, 'First', color=axcolor, hovercolor='0.975')
        self.button_first.on_clicked(self.first_episode_num)
        self.button_last = Button(axlast, 'Last', color=axcolor, hovercolor='0.975')
        self.button_last.on_clicked(self.last_episode_num)

        # Next/Prev button
        self.button_prev = Button(axprev, 'Prev', color=axcolor, hovercolor='0.975')
        self.button_prev.on_clicked(self.prev_episode_num)
        self.button_next = Button(axnext, 'Next', color=axcolor, hovercolor='0.975')
        self.button_next.on_clicked(self.next_episode_num)

        # Timer
        self.timer = self.fig.canvas.new_timer(interval=1000)
        self.timer.add_callback(self.check_update)
        self.timer.start()

        # Progress data
        self.axprogress.set_xmargin(0)
        self.axprogress.set_xlabel('Episodes')
        self.axprogress.set_ylabel('Reward')
        self.axprogress.grid(True)
        self.plot_progress()

        # Plot latest episode
        self.update_episode(self.num_episodes - 1)

        plt.show()

    def check_update(self):
        if self.read_monitor_file():
            self.plot_progress()

    def plot_progress(self):
        # Redraw all lines
        self.axprogress.lines = []
        self.axprogress.plot(self.reward, color='#1f77b4', label='Reward')
        #self.axprogress.plot(self.reward_mean, color='#ff7f0e', label='Reward (average)')
        self.axprogress.legend()
        # Redraw slider
        if self.sl_episode is None or int(round(self.sl_episode.val)) == self.num_episodes - 2:
            cur_ep = self.num_episodes - 1
        else:
            cur_ep = int(round(self.sl_episode.val))
        self.axslider.clear()
        #self.sl_episode = Slider(self.axslider, 'Episode (0..{})'.format(self.num_episodes - 1), 0, self.num_episodes - 1, valinit=self.num_episodes - 1, valfmt='%6.0f')
        self.sl_episode = Slider(self.axslider, 'Episode (0..{})'.format(self.num_episodes - 1), 0, self.num_episodes - 1, valinit=cur_ep, valfmt='%6.0f')
        self.sl_episode.on_changed(self.set_episode_num)

    def read_monitor_file(self):
        # For the very first call, Wait until monitor.csv is created
        if self.timestamp_csv is None:
            while not os.path.isfile(self.monitor_file):
                time.sleep(1)
            self.timestamp_csv = os.stat(self.monitor_file).st_mtime - 1 # '-1' is a hack to prevent losing the first set of data

        num_ep = 0
        ts = os.stat(self.monitor_file).st_mtime
        if ts > self.timestamp_csv:
            # Monitor file is updated.
            self.timestamp_csv = ts
            f = open(self.monitor_file)
            firstline = f.readline()
            assert firstline.startswith('#')
            metadata = json.loads(firstline[1:])
            assert metadata['env_id'] == "EnergyPlus-v0"
            assert set(metadata.keys()) == {'env_id', 't_start'},  "Incorrect keys in monitor metadata"
            df = pd.read_csv(f, index_col=None)
            assert set(df.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
            f.close()

            self.reward = []
            self.reward_mean = []
            self.episode_dirs = []
            self.num_episodes = 0
            for rew, len, time_ in zip(df['r'], df['l'], df['t']):
                self.reward.append(rew / len)
                self.reward_mean.append(rew / len)
                self.episode_dirs.append(self.log_dir + '/output/episode-{:08d}'.format(self.num_episodes))
                self.num_episodes += 1
            if self.num_episodes > self.num_episodes_last:
                self.num_episodes_last = self.num_episodes
                return True
        else:
            return False

    def update_episode(self, ep):
        self.plot_episode(ep)

    def set_episode_num(self, val):
        ep = int(round(self.sl_episode.val))
        self.update_episode(ep)

    def first_episode_num(self, val):
        self.sl_episode.set_val(0)

    def last_episode_num(self, val):
        self.sl_episode.set_val(self.num_episodes - 1)

    def prev_episode_num(self, val):
        ep = int(round(self.sl_episode.val))
        if ep > 0:
            ep -= 1
            self.sl_episode.set_val(ep)

    def next_episode_num(self, val):
        ep = int(round(self.sl_episode.val))
        if ep < self.num_episodes - 1:
            ep += 1
            self.sl_episode.set_val(ep)

    def show_statistics(self, title, series):
        print('{:25} ave={:5,.2f}, min={:5,.2f}, max={:5,.2f}, std={:5,.2f}'.format(title, np.average(series), np.min(series), np.max(series), np.std(series)))

    def get_statistics(self, series):
        return np.average(series), np.min(series), np.max(series), np.std(series)

    def show_distrib(self, title, series):
        dist = [0 for i in range(1000)]
        for v in series:
            idx = int(math.floor(v * 10))
            if idx >= 1000:
                idx = 999
            if idx < 0:
                idx = 0
            dist[idx] += 1
        print(title)
        print('    degree 0.0-0.9 0.0   0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9')
        print('    -------------------------------------------------------------------------')
        for t in range(170, 280, 10):
            print('    {:4.1f}C {:5.1%}  '.format(t / 10.0, sum(dist[t:(t+10)]) / len(series)), end='')
            for tt in range(t, t + 10):
                print(' {:5.1%}'.format(dist[tt] / len(series)), end='')
            print('')

    def get_episode_list(self, log_dir='', csv_file=''):
        if (log_dir is not '' and csv_file is not '') or (log_dir is '' and csv_file is ''):
            print('Either one of log_dir or csv_file must be specified')
            quit()
        if log_dir is not '':
            if not os.path.isdir(log_dir):
                print('energyplus_model.dump: {} is not a directory'.format(log_dir))
                return
            print('energyplus_plot.dump: log={}'.format(log_dir))
            #self.log_dir = log_dir

            # Make a list of all episodes
            # Note: Somethimes csv file is missing in the episode directories
            # We accept gziped csv file also.
            csv_list = glob(log_dir + '/output/episode-????????/eplusout.csv') \
                       + glob(log_dir + '/output/episode-????????/eplusout.csv.gz')
            self.episode_dirs = list(set([os.path.dirname(i) for i in csv_list]))
            self.episode_dirs.sort()
            self.num_episodes = len(self.episode_dirs)
        else: #csv_file != ''
            self.episode_dirs = [ os.path.dirname(csv_file) ]
            self.num_episodes = len(self.episode_dirs)

    # Model dependent methods
    @abstractmethod
    def read_episode(self, ep): pass

    @abstractmethod
    def plot_episode(self, ep): pass

    @abstractmethod
    def dump_timesteps(self, log_dir='', csv_file='', **kwargs): pass

    @abstractmethod
    def dump_episodes(self, log_dir='', csv_file='', **kwargs): pass
