## Docker image usage instructions

The docker image that can be built using `Dockerfile` allows to automate setup and ensure reproducible work 
environment.

### Building

You need `docker` installed. Present procedure was built and tested on Ubuntu 18.04.

Build image with:

```shell
docker build . -t rl-testbed-for-energyplus
```

The default EnergyPlus version used in the image is `9.2.0`. To use a different one, follow example below:

```shell
docker build . \
  --build-arg EPLUS_VERSION=8.8.0 \
  --build-arg EPLUS_DL_URL="<EnergyPlus download URL>" \
  -t rl-testbed-for-energyplus
```

### Running

The image can be used to run a training as it contains all necessary dependencies. For example, if you want to use 
your own project sources to launch a training, you can first start docker image with project sources mounted as a 
container volume:

```shell
docker run -t -i -v /path/to/project:/root/my-project rl-testbed-for-energyplus
```

Tip: simply use `$(pwd)` instead of `/path/to/project` if you want to mount the current directory.

This starts the container and opens a `bash` shell. Then you can launch a training as documented in README.md at the root 
of this project. All modifications you will make to your sources located on your hard drive will be reflected in running 
docker container.
