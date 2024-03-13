
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

LABEL maintainer="mei"
ARG PROJECT_NAME=ml
ARG USER_NAME=docker
ARG GROUP_NAME=docker
ARG UID=1000
ARG GID=1000
ARG PYTHON_VERSION=3.10
ARG APPLICATION_DIRECTORY=/home/${USER_NAME}/${PROJECT_NAME}
ARG RUN_RYE_INSTALL_AT_BUILD_TIME="false"

ENV DEBIAN_FRONTEND="noninteractive" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    TZ=Asia/Tokyo \
    PYTHONPATH=${APPLICATION_DIRECTORY} \
    PYTHONUTF8=1

RUN apt update && apt install --no-install-recommends -y \
    sudo git zip unzip curl wget htop vim tmux make ssh openssh-client fzf build-essential clang \
    python${PYTHON_VERSION} python3-pip python-is-python3

RUN apt-get update && apt-get install -qqy npm

RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Following is needed to swtich default python3 version
# For detail, please check following link https://unix.stackexchange.com/questions/410579/change-the-python3-default-version-in-ubuntu
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    # `requests` needs to be upgraded to avoid RequestsDependencyWarning
    # ref: https://stackoverflow.com/questions/56155627/requestsdependencywarning-urllib3-1-25-2-or-chardet-3-0-4-doesnt-match-a-s
    && python3 -m pip install --upgrade pip setuptools requests

# Add user. Without this, following process is executed as admin.
RUN groupadd -g ${GID} ${GROUP_NAME} \
    && useradd -ms /bin/sh -u ${UID} -g ${GID} ${USER_NAME}
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo 'docker ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ${USER_NAME}
WORKDIR ${APPLICATION_DIRECTORY}

RUN git clone --depth=1 https://github.com/mei28/dotfiles.git ~/dotfiles && rm -rf ~/dotfiles/.git
RUN bash ~/dotfiles/setup.sh

# rye
# ryeをインストール
RUN curl -sSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" bash

# ryeのパス設定
RUN echo 'source "$HOME/.rye/env"' >> ~/.bashrc

# 一つのRUN命令で連続してコマンドを実行（パスを適用し、ryeを使用）
COPY --chown=${UID}:${GID} pyproject.toml .python-version .
RUN /bin/bash -c "source ~/.bashrc && rye pin ${PYTHON_VERSION} && rye sync"
