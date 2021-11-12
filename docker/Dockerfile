FROM nvcr.io/nvidia/pytorch:20.12-py3

# add user and his password
ARG USER=docker_yolo9000
ARG UID=1000
ARG GID=1000

# default password
ARG PW=user

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo
WORKDIR /home/${USER}
USER ${UID}:${GID}

RUN python -m pip install --upgrade pip && \
    pip3 install tensorboardX
