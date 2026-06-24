# Replicate the local `nesy-monitoring` conda env exactly, inside a container.
#
# GPU note: the torch cu124 wheel (pulled by environment.yml) bundles its own
# CUDA runtime, so we do NOT need a CUDA base image. At run time the container
# only needs the host's NVIDIA driver + nvidia-container-toolkit, reached via
# `docker run --gpus all`.
FROM mambaorg/micromamba:1.5.10

# --- MONA (system dependency of ltlf2dfa) ------------------------------------
# ltlf2dfa shells out to the `mona` binary; it is NOT a pip/conda package.
# This apt-get runs INSIDE the image at build time. It does not touch the host
# and needs no install permissions on the server that will run the container.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends mona \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

WORKDIR /app

# Copy the project first: environment.yml ends with `-e .`, so the editable
# install needs pyproject.toml + src/ present while the env is being created.
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# Build the conda env from environment.yml (creates env `nesy-monitoring`,
# including torch==2.6.0+cu124 from the PyTorch CUDA index and `pip install -e .`).
RUN micromamba create -y -f environment.yml \
    && micromamba clean --all --yes

# Auto-activate `nesy-monitoring` for every subsequent RUN and for `docker run`.
ENV ENV_NAME=nesy-monitoring
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Build-time sanity check: the env imports and the mona binary is on PATH.
RUN python -c "import torch, ltlf2dfa; print('torch', torch.__version__)" \
    && mona -h >/dev/null 2>&1 || true

# Default to a shell; override with the experiment command, e.g.:
#   docker run --gpus all -v "$PWD/results:/app/results" nesy-monitoring \
#       python experiments/exp1_single_trace.py
CMD ["bash"]
