# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from gym.envs.registration import register

register(
    id='EnergyPlus-v0',
    entry_point='gym_energyplus.envs:EnergyPlusEnv',
)
