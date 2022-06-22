FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libosmesa6-dev \
    patchelf \
    python3-opengl \
    python3-dev=3.8* \
    python3-pip \
    sudo \
    unzip \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install (mini)conda. This will enable us to install python packages
# for all users without overriding or impacting in any way the python packages installed by
# root. This is important because the user we will end up using is parametrized by a dockerfile
# argument (see USER_ID later in this file) and we want to share the docker image cache as much
# as possible for all possible values of USER_ID.
RUN curl https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc | gpg --dearmor > conda.gpg
RUN install -o root -g root -m 644 conda.gpg /usr/share/keyrings/conda-archive-keyring.gpg
RUN gpg --keyring /usr/share/keyrings/conda-archive-keyring.gpg --no-default-keyring --fingerprint 34161F5BF5EB1D4BFBBB8F0A8AEB4F8B29D82806
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" > /etc/apt/sources.list.d/conda.list

RUN apt-get update && \
    apt-get install -y --no-install-recommends conda=4.9.2-0 && \
    rm -rf /var/lib/apt/lists/*

# Install all pip dependencies through conda's global pip.
# Also manually list here all mujoco_py pip dependencies before installing
# mujoco_py itself as doing otherwise leads to errors (only observed for mujoco_py < 2.*)
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/targets/x86_64-linux/lib
ENV PATH=/opt/conda/bin:$PATH

COPY ./requirements.txt /tmp/requirements.txt

RUN pip3 --no-cache-dir install -r /tmp/requirements.txt \
    jaxlib==0.1.75+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html \
    && rm -rf /tmp/*

# Create 'eng' user (unfortunately mujoco 1.5 needs to be installed by a user - not root)
# The id and group-id of 'eng' can be parametrized to match that of the user that will use this
# docker image so that the eng can create files in mounted directories seamlessly (without
# permission issues)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid ${GROUP_ID} eng
RUN useradd -l --gid eng --uid ${USER_ID} --shell /bin/bash --home-dir /app --create-home eng
WORKDIR /app
USER eng

# Install mujoco 1.50 (as opposed to 2.*) because versions 2.*
# are not officially supported by openai gym yet and bugs
# have been reported, see https://github.com/openai/gym/issues/1541
RUN mkdir .mujoco
RUN wget --no-check-certificate https://www.roboti.us/download/mjpro150_linux.zip
RUN unzip mjpro150_linux.zip && mv mjpro150 .mujoco/mjpro150 && rm mjpro150_linux.zip
RUN cd .mujoco/ && wget --no-check-certificate https://roboti.us/file/mjkey.txt

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app/.mujoco/mjpro150/bin

RUN pip3 --no-cache-dir install mujoco_py==1.50.1.68

ENV PATH=/app/bin:/app/.local/bin:$PATH

# Disable debug, info, and warning tensorflow logs
ENV TF_CPP_MIN_LOG_LEVEL=3
# By default use cpu as the backend for JAX, we will
# explicitely load data on gpus as needed.
ENV JAX_PLATFORM_NAME="cpu"

# Create fastpbrl where the repository is expected to be mounted
RUN mkdir fastpbrl

# Add symlink to fastpbrl python package so that users do not have to
# run scripts with the python -m option.
USER root
RUN ln -s /app/fastpbrl/fastpbrl /opt/conda/lib/python3.8/site-packages/fastpbrl
USER eng
WORKDIR /app/fastpbrl
