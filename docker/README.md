## Docker image usage instructions

The docker image that can be built using `Dockerfile` allows automating setup and ensures reproducible work 
environment.

### Pre-built images

See the existing [list of docker images](https://github.com/users/antoine-galataud/packages/container/package/rl-testbed-for-energyplus)

Usage example:

```shell
docker pull ghcr.io/antoine-galataud/rl-testbed-for-energyplus:ray1.12.1_ep22.1.0
```

where image tag `ray1.12.1_ep22.1.0` are Ray RLlib and EnergyPlus versions the image was built with.

### Building

You need `docker` installed (ie `sudo apt install docker.io`).

Build image with:

```shell
cd rl-testbed-for-energyplus/
docker build . -f docker/Dockerfile -t rl-testbed-for-energyplus
```

The default EnergyPlus version used in the image is `22-1-0`. The default RL framework is OpenAI baselines.

To use different options, follow example below:

```shell
docker build . \
  --build-arg EPLUS_VERSION=9-6-0 \
  --build-arg EPLUS_DL_URL="<EnergyPlus download URL>" \
  --build-arg RL_FRAMEWORK="ray" \
  -t rl-testbed-for-energyplus
```

### Running

The image can be used to run a training as it contains all necessary dependencies. Run:

```shell
docker run -t -i rl-testbed-for-energyplus
```

to start the container and open a shell. Then: 

```shell
cd /root/rl-testbed-for-energyplus
# launch (OpenAI baselines)
time python3 -m baselines_energyplus.trpo_mpi.run_energyplus --num-timesteps 1000000000
# launch (Ray RLlib - PPO)
time python3 -m ray_energyplus.ppo.run_energyplus --num-timesteps 1000000000
```

Another option is to use your own project sources to launch a training (e.g. you have forked present git repo). To do 
so, you can start docker image with your project sources mounted as a container volume:

```shell
docker run -t -i -v /path/to/project:/root/my-project rl-testbed-for-energyplus
```

Tip: simply use `$(pwd)` instead of `/path/to/project` if you want to mount the current directory.

Note: all modifications you will make to your sources located on your hard drive will be reflected in the running 
docker container.

### Plotting

If you want to monitor progress using plots from inside the docker container, this can be done in 3 steps (tested on Ubuntu, probably not working on MacOS):

Make sure all clients can access your display:

```shell
sudo apt install x11-xserver-utils
xhost +
```

Run the container so it can access your display:

```shell
docker run -it \
    --env="QT_X11_NO_MITSHM=1" \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    rl-testbed-for-energyplus
```

Run the experiment, then run monitoring tool as usual (inside the container):

```shell
python3 -m common.plot_energyplus
```

For more information, see also [this tutorial](http://wiki.ros.org/docker/Tutorials/GUI)
