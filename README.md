# Project Description
Reinforcement Learning Testbed for Power Consumption Optimization.

## Contributing to the project
We welcome contributions to this project in many forms. There's always plenty to do! Full details of how to contribute to this project are documented in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Maintainers
The project's [maintainers](MAINTAINERS.txt): are responsible for reviewing and merging all pull requests and they guide the over-all technical direction of the project.

## Supported platforms
We have tested on the following platforms.
- macOS High Sierra (Version 10.13.6)
- Ubuntu 16.04.2 LTS

## Installation
Installaton of rl-testbed-for-energyplus consists of three parts:

- Intall EnergyPlus prebuild package
- Build patched EnergyPlus
- Install built executables

### Install EnergyPlus prebuild package
First, download pre-built package of EnergyPlus and install it.
This is not for executing normal version of EnergyPlus, but to get some pre-compiled binaries and data files that can not be generated from source code.
You need to download ver. 8.8.0 from https://github.com/NREL/EnergyPlus/releases/tag/v8.8.0.
Newer version (v8.9.0) is not suppoted yet.

#### Ubuntu

1. Go to the web page shown above.
2. Right click
   [EnergyPlus-8.8.0-7c3bbe4830-Linux-x86_64.sh](https://github.com/NREL/EnergyPlus/releases/download/v8.8.0/EnergyPlus-8.8.0-7c3bbe4830-Linux-x86_64.sh) and select `Save link As` to from the menu to download installation image.
3. Execute installation image.
```
$ sudo bash <DOWNLOAD-DIRECTORY>/EnergyPlus-8.8.0-7c3bbe4830-Linux-x86_64.sh
```
Enter your admin password if required.
Specify `/usr/local` for install directory.
Respond with `/usr/local/bin` if asked for symbolic link location.
The package will be installed at `/usr/local/EnergyPlus-8-8-0`.

#### macOS

1. Go to the web page shown above.
2. Right click
   [EnergyPlus-8.8.0-7c3bbe4830-Darwin-x86_64.dmg](https://github.com/NREL/EnergyPlus/releases/download/v8.8.0/EnergyPlus-8.8.0-7c3bbe4830-Darwin-x86_64.dmg) and select Save link As` to from the menu to download installation image.
3. Double click the downloaded package, and follow the instructions.
The package will be installed in `/Applications/EnergyPlus-8-8-0`.

### Build patched EnergyPlus

Download source code of EnergyPlus and rl-testbed-for-energyplus.

```
$ cd <WORKING-DIRECTORY>
$ git clone -b v8.8.0 git@github.com:NREL/EnergyPlus.git
$ git clone git@github.com:ibm/rl-testbed-for-energyplus.git
```

Apply patch to EnergyPlus and build.

```
$ cd <WORKING-DIRECTORY>/EnergyPlus
$ patch -p1 < ../rl-testbed-for-energyplus/EnergyPlus/RL-patch-for-EnergyPlus-8-8-0.patch
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=/usr/local/EnergyPlus-8-8-0 ..    # Ubuntu case
$ cmake -DCMAKE_INSTALL_PREFIX=/Applications/EnergyPlus-8-8-0 .. # macOS case
$ make -j4
```

### Install built executables
```
$ sudo make install
```

### Install OpenAI Gym and OpenAI Baselines
See https://github.com/openai/baselines for details.

```
$ pip3 install gym
$ pip3 install baselines
```

## How to run

### Set up
Some environment variables must be defined.

In `$(HOME)/.bashrc`
```
# Specify the top directory
TOP=SOMEWHERE/rl-testbed-for-energyplus
export PYTHONPATH=${PYTHONPATH}:${TOP}
MODEL_DIR="${TOP}/EnergyPlus/Model"

if [ `uname` == "Darwin" ]; then
	energyplus_instdir="/Applications"
else
	energyplus_instdir="/usr/local"
fi
ENERGYPLUS_VERSION="8-8-0"
ENERGYPLUS_DIR="${energyplus_instdir}/EnergyPlus-${ENERGYPLUS_VERSION}"
WEATHER_DIR="${ENERGYPLUS_DIR}/WeatherData"
export ENERGYPLUS="${ENERGYPLUS_DIR}/energyplus"

# Weather file.
# Single weather file or multiple weather files separated by comma character.
export ENERGYPLUS_WEATHER="${WEATHER_DIR}/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
#export ENERGYPLUS_WEATHER="${WEATHER_DIR}/USA_CO_Golden-NREL.724666_TMY3.epw"
#export ENERGYPLUS_WEATHER="${WEATHER_DIR}/USA_FL_Tampa.Intl.AP.722110_TMY3.epw"
#export ENERGYPLUS_WEATHER="${WEATHER_DIR}/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
#export ENERGYPLUS_WEATHER="${WEATHER_DIR}/USA_VA_Sterling-Washington.Dulles.Intl.AP.724030_TMY3.epw"
#export ENERGYPLUS_WEATHER="${WEATHER_DIR}/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw,${WEATHER_DIR}/USA_CO_Golden-NREL.724666_TMY3.epw,${WEATHER_DIR}/USA_FL_Tampa.Intl.AP.722110_TMY3.epw"

# Ouput directory "openai-YYYY-MM-DD-HH-MM-SS-mmmmmm" is created in
# the directory specified by ENERGYPLUS_LOGBASE or in the current directory if not specified.
export ENERGYPLUS_LOGBASE="${HOME}/eplog"

# Model file. Uncomment one.
#export ENERGYPLUS_MODEL="${MODEL_DIR}/2ZoneDataCenterHVAC_wEconomizer_Temp.idf"     # Temp. setpoint control
export ENERGYPLUS_MODEL="${MODEL_DIR}/2ZoneDataCenterHVAC_wEconomizer_Temp_Fan.idf" # Temp. setpoint and fan control

# Run command (example)
# $ time python3 -m baselines_energyplus.trpo_mpi.run_energyplus --num-timesteps 1000000000

# Monitoring (example)
# $ python3 -m baselines_energyplus.common.plot_energyplus
```

### Running
Simulation process starts by the following command. The only applicable option is `--num-timesteps`

```
$ time python3 -m baselines_energyplus.trpo_mpi.run_energyplus --num-timesteps 1000000000
```
Output files are generated under the directory `${ENERGYPLUS_LOGBASE}/openai-YYYY-MM-DD-HH-MM-SS-mmmmmm`. These include:
- log.txt       Log file generated by baselines Logger.
- progress.csv  Log file generated by baselines Logger.
- output/episode-NNNNNNNN/ Episode data

Epsiode data contains the following files:
- 2ZoneDataCenterHVAC_wEconomizer_Temp_Fan.idf  A copy of model file used in the simulation of the episode
- USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw  A copy of weather file used in the simulation of the episode
- eplusout.csv.gz                               Simulation result in CSV format
- eplusout.err                                  Error message. You need make sure that there are no Severe errors
- eplusout.htm                                  Human readable report file

### Monitoring

You can monitor the progress of the simulation using plot_energyplus utility.

```
$ python3 -m baselines_energyplus.common.plot_energyplus
Options:
- -l <log_dir>    Specify log directory (usually openai-YYYY-MM-DD-HH-MM-SS-mmmmmm)
- -c <csv_file>   Specify single CSV file to view
- -d              Dump every timestep in CSV file (dump_timesteps.csv)
- -D              Dump episodes in CSV file (dump_episodes.dat)
```

If neither `-l` nor `-c` option is specified, plot_energyplus tries to open the latest directory under `${ENERGYPLUS_LOG}` directory.
If none of `-d` or `-D` is specified, the progress windows is opened.

![EnergyPlus monitor](/images/energyplus_plot.png)

Several graphs are shown.
1. Zone temperature and outdoor temperature
2. West zone return air temperature and west zone setpoint temperature
3. Mixed air, fan, and DEC outlet temperatures
4. IEC, CW, DEC outlet temperatures
5. Electric demand power (whole building, facility, HVAC)
6. Reward

Only the curent episode is shown in the graph 1 to 5. The current episode is specified by pushing one of "First", "Prev", "Next", or
"Last" buton, or directly clicking the appropriate point on the episode bar at the bottom.
If you're at the last episode, the current episode moves automatically to the latest one as new episode is completed.

Note: The reward value shown in the graph 6 is retrieved from "progress.csv" file generated by TRPO baseline, which is not
necessarily same as the reward value computed by our reward function.

You can pan or zoom each graph by entering pan/zoom mode by clicking cross-arrows on the bottom left of the window.

When new episode is shown on the window, some statistical information is show as follow:

```
episode 362
read_episode: file=/home/moriyama/eplog/openai-2018-07-04-10-48-46-712881/output/episode-00000362/eplusout.csv.gz
Reward                    ave= 0.77, min= 0.40, max= 1.33, std= 0.22
westzone_temp             ave=22.93, min=21.96, max=23.37, std= 0.19
eastzone_temp             ave=22.94, min=22.10, max=23.51, std= 0.17
Power consumption         ave=102,243.47, min=65,428.31, max=135,956.47, std=18,264.50
pue                       ave= 1.27, min= 1.02, max= 1.63, std= 0.13
westzone_temp distribution
    degree 0.0-0.9 0.0   0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9
    -------------------------------------------------------------------------
    18.0C  0.0%    0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%
    19.0C  0.0%    0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%
    20.0C  0.0%    0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%
    21.0C  0.0%    0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%
    22.0C 50.8%    0.0%  0.1%  0.7%  3.4%  5.6%  2.2%  0.7%  1.0%  0.9% 36.4%
    23.0C 49.2%   49.0%  0.2%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%
    24.0C  0.0%    0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%
    25.0C  0.0%    0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%
    26.0C  0.0%    0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%
    27.0C  0.0%    0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%  0.0%
```
The reward value shown above is computed by applying reward function to the simulation result.

## License <a name="license"></a>
The Reinforcement Learning Testbed for Power Consumption Optimization Project uses the [MIT License](LICENSE) software license.

## Related information
- AsiaSim2018: http://jsst.jp/asiasim2018 - A technical paper on this project will be presented at this conference.
- A pre-print version of AsiaSim2018 paper on arXiv: https://arxiv.org/abs/1808.10427
- EnergyPlus: https://github.com/NREL/EnergyPlus
- OpenAI Gym: https://github.com/OpenAI/gym
- OpenAI Baselines: https://github.com/OpenAI/baselines
