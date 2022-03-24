## Docker image usage instructions

The docker image that can be built using `Dockerfile` allows automating setup and ensures reproducible work 
environment.

### Building

You need `docker` installed (ie `sudo apt install docker.io`).

Build image with:

```shell
cd rl-testbed-for-energyplus/
docker build . -f docker/Dockerfile -t rl-testbed-for-energyplus
```

The default EnergyPlus version used in the image is `9-5-0`. To use a different one, follow example below:

```shell
docker build . \
  --build-arg EPLUS_VERSION=9-4-0 \
  --build-arg EPLUS_DL_URL="<EnergyPlus download URL>" \
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
# launch
`time python3 -m baselines_energyplus.trpo_mpi.run_energyplus --num-timesteps 1000000000`
```

Another option is to use your own project sources to launch a training (e.g. you have forked present git repo). To do 
so, you can start docker image with your project sources mounted as a container volume:

```shell
docker run -t -i -v /path/to/project:/root/my-project rl-testbed-for-energyplus
```

Tip: simply use `$(pwd)` instead of `/path/to/project` if you want to mount the current directory.

Note: all modifications you will make to your sources located on your hard drive will be reflected in the running 
docker container.
