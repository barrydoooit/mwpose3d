# Installation Guide

## Prerequisites

- Linux (and Jetson) / Windows supported
- Python
- git
- uv

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.cargo/env"
uv --version
```

## Clone repositories

Clone both repos side-by-side:

```bash
git clone https://github.com/barrydoooit/mwCore.git --branch mwcore-dev
git clone https://github.com/barrydoooit/mwpose3d.git --branch mwpose3d-dev
```

## Install and run (recommended development setup)

### Step 1: Create the environment for mwpose3d

`uv sync` will create/update `.venv` inside the `mwpose3d` folder using `uv.lock`.

```bash
cd mwpose3d
uv sync
```

### Step 2: Activate the environment

```bash
source .venv/bin/activate
```

### Step 3: Install mwpose3d (editable)

From inside the `mwpose3d` repo:

```bash
uv pip install -e .
```

### Step 4: Quick import test

```bash
python -c "import mwcore, mwpose3d; print('OK')"
```

## Jetson note: PyTorch index

On Jetson, `torch` and `torchvision` often must be installed from a Jetson-compatible wheel index. For example, for JetPack 6.x with CUDA 12.6, run:

```bash
uv pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

Then rerun `uv sync` to ensure all other dependencies are met:

```bash
uv sync
```

Verification:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"
```