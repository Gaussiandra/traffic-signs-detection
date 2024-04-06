FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    wget \
    curl \
    git \
    vim \
    openssh-client \
    build-essential \
    zip \
    unzip \
    ffmpeg \
    libsm6 \
    libxext6

# install Poetry
# following https://python-poetry.org/docs/#ci-recommendations
ENV POETRY_VERSION=1.2.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

RUN python3.9 -m venv $POETRY_VENV \
  && $POETRY_VENV/bin/pip install -U pip setuptools \
  && $POETRY_VENV/bin/pip install poetry==$POETRY_VERSION

ENV PATH="${PATH}:${POETRY_VENV}/bin"
RUN poetry config virtualenvs.create false

# install conda
# same as https://hub.docker.com/r/continuumio/miniconda3
ENV CONDA_VERSION=py39_4.12.0
ENV CONDA_PATH=/opt/conda

RUN export UNAME_M="$(uname -m)" \
  && wget -O /tmp/miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-${UNAME_M}.sh \
  && bash /tmp/miniconda3.sh -b -p $CONDA_PATH
RUN ln -s ${CONDA_PATH}/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
  && echo ". ${CONDA_PATH}/etc/profile.d/conda.sh" >> ~/.bashrc \
  && echo "conda activate base" >> ~/.bashrc \
  && find ${CONDA_PATH}/ -follow -type f -name '*.a' -delete \
  && find ${CONDA_PATH}/ -follow -type f -name '*.js.map' -delete \
  && ${CONDA_PATH}/bin/conda clean -afy

SHELL ["/bin/bash", "--login", "-c"]

ARG PYTHON_VERSION=3.10
ARG CONDA_ENV=dev

RUN conda create -n $CONDA_ENV python=$PYTHON_VERSION
RUN conda activate dev

# Install project dependencies
COPY pyproject.toml poetry.lock ./
RUN conda run -n $CONDA_ENV poetry install

WORKDIR /workspace
