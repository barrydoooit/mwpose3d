# Installation Guide

## Prerequisites

- Ubuntu / Windows supported
- Python **3.10–3.12** (recommended)
- git
- uv

> Note: PyTorch wheels are published per Python version and platform. You must use a PyTorch version that provides **cp3xx** wheels for your selected CUDA build.

## Install uv

### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.cargo/env"
uv --version
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

## Clone repositories

Clone both repos side-by-side:

```bash
git clone https://github.com/barrydoooit/mwCore.git --branch mwcore-dev
git clone https://github.com/barrydoooit/mwpose3d.git --branch mwpose3d-dev
cd mwpose3d
```

## Install and run (recommended development setup)

### Step 1: Create the environment

Pick **one** PyTorch variant (extras are mutually exclusive):

- `cpu`   → CPU-only wheels
- `cu{xx}{y}` → CUDA xx.y  wheels

For example, when installing for CUDA 12.4:
#### Linux and Windows (PowerShell)

```bash
uv sync --extra cu124
```

### Step 2: Activate the environment

#### Linux

```bash
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
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

### Step 5: Verify PyTorch + CUDA (if using a CUDA extra)

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
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