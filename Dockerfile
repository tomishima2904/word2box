FROM azraelkuan/pytorch1.8.0-hvd-apex-py38-cuda10.2-cudnn8
USER root
# add user
ARG USERNAME=tomishima2904
ARG USER_UID=1098
ARG USER_GID=1000

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm
ENV PYTHONIOENCODING utf-8

COPY requirements.txt /mnt/tomishima2904/word2box/
WORKDIR /mnt/tomishima2904/word2box/

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# To handle error below
# InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
#RUN apt-key del 3bf863cc
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install python3 python3-pip
RUN apt-get -y install nano wget curl vim

WORKDIR /mnt/tomishima2904/word2box/

# Delete line below if you want to play as root
USER $USERNAME