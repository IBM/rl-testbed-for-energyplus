ARG EPLUS_VERSION=9-2-0
ARG EPLUS_DL_URL=https://github.com/NREL/EnergyPlus/releases/download/v9.2.0/EnergyPlus-9.2.0-921312fa1d-Linux-x86_64.sh

# Stage 1
FROM ubuntu:18.04 AS eplus-build

ARG EPLUS_VERSION
ARG EPLUS_DL_URL
ARG EPLUS_BUILD_PARALLELISM=4
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

COPY ./EnergyPlus /root/patches/
WORKDIR /root

RUN apt update && \
    apt install -y -u build-essential cmake python3-minimal wget openssh-client git libgl1-mesa-glx && \
    # necessary so cmake can find python binary
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
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
FROM ubuntu:18.04

ARG EPLUS_VERSION

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
    apt update && apt install -y locales python3-pip python3-mpi4py git libgl1-mesa-glx && \
    pip3 install -U pip && \
    # installing deps in 2 steps:
    # - baselines 0.1.5 brings mujoco (not needed here), so we have to use 0.1.6 (which has no wheel, so from git)
    # - baselines requires tensorflow installed first
    cd /root/rl-testbed-for-energyplus && \
    cat requirements.txt | grep -v baselines | xargs pip3 install && \
    cat requirements.txt | grep baselines | xargs pip3 install && \
    echo "source /root/.bashrc_eplus" >> /root/.bashrc && \
    # cleanup
    apt autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean

ENTRYPOINT "/bin/bash"
