ARG EPLUS_VERSION=9-5-0
ARG EPLUS_DL_URL=https://github.com/NREL/EnergyPlus/releases/download/v9.5.0/EnergyPlus-9.5.0-de239b2e5f-Linux-Ubuntu20.04-x86_64.sh

# Stage 1
FROM ubuntu:20.04 AS eplus-build

ARG EPLUS_VERSION
ARG EPLUS_DL_URL
ARG EPLUS_BUILD_PARALLELISM=4
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

COPY ./EnergyPlus /root/patches/
WORKDIR /root

RUN apt update && \
    apt install -y -u build-essential cmake python3-minimal wget openssh-client git libgl1-mesa-glx && \
    # install EnergyPlus from binaries
    wget --quiet $EPLUS_DL_URL && \
    (echo "y"; echo ""; echo "";) | bash $(echo "$EPLUS_DL_URL" | rev | cut -d'/' -f1 | rev) && \
    # get EnergyPlus sources
    ssh-keyscan -H github.com >> /etc/ssh/ssh_known_hosts && \
    git clone --depth=1 -b v${EPLUS_VERSION//-/.} https://github.com/NREL/EnergyPlus.git && \
    # build patched EnergyPlus
    export LC_ALL=C.UTF-8 && cd /root/EnergyPlus && \
    patch -p1 < /root/patches/RL-patch-for-EnergyPlus-${EPLUS_VERSION}.patch && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/EnergyPlus-${EPLUS_VERSION} .. && \
    make -j$EPLUS_BUILD_PARALLELISM && \
    make install

# Stage 2
FROM ubuntu:20.04

ARG EPLUS_VERSION
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# copy file with environment variables. They will be loaded automatically
COPY docker/bashrc_eplus /root/.bashrc_eplus
# copy current sources
COPY ./ /root/rl-testbed-for-energyplus
# copy built EnergyPlus binaries from stage 1
COPY --from=eplus-build \
     /usr/local/EnergyPlus-${EPLUS_VERSION} \
     /usr/local/EnergyPlus-${EPLUS_VERSION}

RUN export EPLUS_INSTALL_PATH=/usr/local/EnergyPlus-${EPLUS_VERSION} && \
    # setup important symbolic links for EnergyPlus (they are not preserved from stage 1)
    ln -s ${EPLUS_INSTALL_PATH}/energyplus /usr/local/bin/energyplus && \
    ln -s ${EPLUS_INSTALL_PATH}/Energy+.idd /usr/local/bin/Energy+.idd && \
    ln -s ${EPLUS_INSTALL_PATH}/ExpandObjects /usr/local/bin/ExpandObjects && \
    ln -s ${EPLUS_INSTALL_PATH}/ReadVarsESO /usr/local/bin/ReadVarsESO && \
    # install python dependencies
    apt update && apt install -y locales python3-pip python3-mpi4py git libgl1-mesa-glx libglib2.0-0 && \
    pip3 install -U pip && \
    # installing deps in 2 steps:
    # - baselines requires tensorflow installed first
    # - we need baselines to support tf2, which is only available in a git branch
    cd /root/rl-testbed-for-energyplus && \
    cat requirements.txt | grep -v baselines | xargs pip3 install --ignore-installed && \
    # install baselines sources in /root/src/baselines. They need to be added to PYTHONPATH
    cd /root && \
    cat requirements.txt | grep baselines | xargs pip3 install && \
    echo "source /root/.bashrc_eplus" >> /root/.bashrc && \
    # cleanup
    apt autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean

ENTRYPOINT "/bin/bash"
