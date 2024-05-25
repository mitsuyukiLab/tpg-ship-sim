FROM python:3

USER root
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    sudo \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN python -m pip install numpy \
    distance \
    geopy \
    folium \
    cartopy \
    matplotlib \
    tz \
    datetime timedelta  \
    polars==0.19.2 \
    selenium \
    SciencePlots \
    opencv-python \
    jupyterlab

# Linter and Formatter
RUN pip install flake8 black isort

# hydra-core
RUN pip install hydra-core --upgrade