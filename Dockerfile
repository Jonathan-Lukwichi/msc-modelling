# Reproduces the Chapter 6 modelling pipeline without host-side Python
# setup. Ships the CORE dependencies only (requirements-core.txt); the
# optional DVC/MLflow/Hydra stack is layered on in docker-compose.yml
# via a second `pip install` so the base image stays lean for anyone
# who only wants to run/verify the pipeline.
#
# Build:  docker build -t msc-modelling .
# Run:    docker run --rm msc-modelling python make.py crossval
# Or use docker-compose.yml for the full multi-service setup (pipeline
# + MLflow UI).

FROM python:3.13-slim

# System packages needed at build time:
#   build-essential + gfortran -> statsmodels/scipy/pmdarima wheels that
#     don't ship prebuilt manylinux wheels for this python/arch combo
#   libgomp1 -> XGBoost's OpenMP runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (own layer) so code edits don't bust the
# dependency cache on rebuild.
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

# MLflow is part of the Tier-1 reproducibility platform (experiment
# tracking for scripts/06, 07, 08, 24, 26, 30) and is small enough to
# ship in the base image; the rest of requirements-optional.txt (Hydra,
# DVC's cloud remotes, the Nixtla stack, DeepAR/gluonts) stays opt-in.
RUN pip install --no-cache-dir "mlflow>=2.9"

# Copy the rest of the project.
COPY . .

# Non-root user (avoids root-owned files leaking into the bind-mounted
# artefacts/ directory on the host).
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Default: print the runbook. Override with `docker run ... python make.py <cmd>`.
CMD ["python", "make.py", "help"]
