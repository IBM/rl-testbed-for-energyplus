# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

import os
from gym_energyplus.envs.Timeout import Timeout

class PipeIo():
    
    EXTCTRL_PIPE_DIR = "/tmp"
    
    def __init__(self):

        self.extctrl_pipe_prefix = self.EXTCTRL_PIPE_DIR + "/extctrl_" + str(os.getpid())
        self.obs_pipe_filename = self.extctrl_pipe_prefix + "_obs"
        self.act_pipe_filename = self.extctrl_pipe_prefix + "_act"

        # Pipe descriptors and seq. counters
        self.act_pipe = None
        self.obs_pipe = None
        self.act_seq = 0
        self.obs_seq = 0

    def stop(self):
        if (self.obs_pipe is not None):
            self.obs_pipe.close()
            self.obs_pipe = None
        try:
            os.unlink(self.obs_pipe_filename)
        except: pass
        
        if (self.act_pipe is not None):
            self.act_pipe.close()
            self.act_pipe = None
        try:
            os.unlink(self.act_pipe_filename)
        except: pass

    def start(self):
        os.mkfifo(self.obs_pipe_filename)
        os.mkfifo(self.act_pipe_filename)
        self.act_seq = 0
        self.obs_seq = 0
        # Environment variables that should be passed to the child process
        os.environ["OBS_PIPE_FILENAME"] = self.obs_pipe_filename
        os.environ["ACT_PIPE_FILENAME"] = self.act_pipe_filename

    def readline(self):
        # Delayed open
        if (self.obs_pipe is None):
            print('PipeIo.readline: Opening OBS pipe [{}]'.format(self.obs_pipe_filename))
            try:
                with Timeout(1200): # Should be long enough to finish one episode for execution of non-RL model
                    self.obs_pipe = open(self.obs_pipe_filename, 'r')
            except Timeout.Timeout:
                print('Opening file {} timed out'.format(self.obs_pipe_filename))
                return ''
        while True:
            line = self.obs_pipe.readline()[:-1]
            if (line == ''):
                return line
            items = line.split(',', 1)
            if (len(items) == 2):
                break
        seq = int(items[0])
        val = float(items[1])
        assert(self.obs_seq == seq)
        self.obs_seq += 1
        return val

    def writeline(self, str):
        # Delayed open
        if (self.act_pipe is None):
            try:
                with Timeout(1):
                    self.act_pipe = open(self.act_pipe_filename, 'w')
            except Timeout.Timeout:
                print('Opening file {} timed out'.format(self.act_pipe_filename))
                return True
            print('PipeIo.writeline: Opened ACT pipe {}'.format(self.act_pipe_filename))
        line = '{0:d},{1:s}'.format(self.act_seq, str)
        self.act_pipe.write('{}\n'.format(line))
        self.act_seq += 1
        return False
        
    def flush(self):
        self.act_pipe.write('DELIMITER\n') # Send no-seq line as delimiter
        self.act_pipe.flush()
