FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
USER root

# 環境変数を定義
# Define env vars
ENV LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja LC_ALL=ja_JP.UTF-8 \
    TZ=JST-9 TERM=xterm PYTHONIOENCODING=utf-8

WORKDIR /mnt/tomishima2904/word2box

# Define the user and the group
ARG USERNAME=tomishima2904 \
    USER_UID=1098 \
    USER_GID=1000

# docker image に上記変数で定義したユーザーが存在しない場合、ユーザーを登録
# If the user defiend above does not exists, add this
RUN if ! id -u $USERNAME > /dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y \
    python3 python3-pip python3-dev python-is-python3 \
    build-essential ca-certificates language-pack-ja \
    nano vim wget curl file git make xz-utils \
    mecab libmecab-dev mecab-ipadic-utf8

# Word2Boxの環境構築
# Dev env of Word2Box
COPY requirements.txt setup.cfg setup.py pyproject.toml fastentrypoints.py ./
COPY lib ./lib/
COPY src/language_modeling_with_boxes/ ./src/language_modeling_with_boxes/
RUN pip install -e lib/*  && pip install -e .

COPY mecab /mnt/tomishima2904/word2box/mecab/

# MeCabのインストール
# Install Mecab
RUN tar zxfv mecab/mecab-0.996.tar.gz && \
    cd mecab-0.996 && \
    ./configure && \
    make && \
    make install && \
    cd ../ && \
    rm -rf mecab-0.996

# MeCab辞書のインストール
# Install the dictionary of MeCab
RUN tar zxfv mecab/mecab-ipadic-2.7.0-20070801.tar.gz && \
    cd mecab-ipadic-2.7.0-20070801 && \
    ./configure && \
    make && \
    make install & \
    cd ../ & \
    rm -rf mecab/mecab-ipadic-2.7.0-20070801

# mecab-ipadic-NEologdのインストール
# Install mecab-ipadic-NEologd
# RUN cd mecab/mecab-ipadic-neologd && \
#     ./bin/install-mecab-ipadic-neologd -n && \
#     cd ../../

# コンテナ内でルートユーザーとしてのみ振る舞いたいなら以下を消す
# Delete line below if you want to play as root
USER $USERNAME
