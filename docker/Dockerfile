FROM ubuntu:18.04

ARG EPLUS_VERSION=9.2.0
ARG EPLUS_DL_URL=https://github.com/NREL/EnergyPlus/releases/download/v9.2.0/EnergyPlus-9.2.0-921312fa1d-Linux-x86_64.sh
ARG EPLUS_BUILD_PARALLELISM=4
ARG RL_TESTBED_BRANCH=master

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root

RUN apt update && \
    apt install -y -u build-essential cmake python3-minimal python3-pip wget openssh-client git && \
    # necessary so cmake can find python binary
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    # install E+ from binaries
    wget --quiet $EPLUS_DL_URL && \
    (echo "y"; echo ""; echo "";) | bash $(echo "$EPLUS_DL_URL" | rev | cut -d'/' -f1 | rev)

# get E+ and rl-testbed-for-energyplussources
RUN ssh-keyscan -H github.com >> /etc/ssh/ssh_known_hosts && \
    git clone --depth=1 -b v$EPLUS_VERSION https://github.com/NREL/EnergyPlus.git && \
    # get rl-testbed-for-energyplus sources
    git clone --depth=1 -b $RL_TESTBED_BRANCH https://github.com/antoine-galataud/rl-testbed-for-energyplus.git ./rl-testbed-for-energyplus

WORKDIR /root/EnergyPlus

# build patched E+
RUN patch -p1 < ../rl-testbed-for-energyplus/EnergyPlus/RL-patch-for-EnergyPlus-$(echo "$EPLUS_VERSION" | tr '.' '-').patch && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/EnergyPlus-$(echo "$EPLUS_VERSION" | tr '.' '-') .. && \
    make -j$EPLUS_BUILD_PARALLELISM && \
    make install

WORKDIR /root

# install python dependencies
RUN apt install -y python3-mpi4py && \
    pip3 install -U pip && \
    cd ./rl-testbed-for-energyplus && \
    # installing deps in 2 steps:
    # - baselines 0.1.5 brings mujoco (not needed here), so we have to use 0.1.6 (which has no wheel, so from git)
    # - baselines requires tensorflow installed first, so have to make sure order is respected (not possible with -r)
    cat requirements.txt | grep -v baselines | xargs pip3 install && \
    cat requirements.txt | grep baselines | xargs pip3 install

# cleanup
RUN rm -rf EnergyPlus && \
    rm $(echo "$EPLUS_DL_URL" | rev | cut -d'/' -f1 | rev) && \
    apt remove -y build-essential cmake wget && \
    apt autoremove -y

ENTRYPOINT "/bin/bash"