# Creating a conda environment

We can replicate a conda environment by loading the `environment.yml` exported from the old environment.

We have three `*.yml` files:
- environment_from_yukuan.yml
    - Received from Yukuan and is a direct copy of his working environment as of 6/6/2025 
- environment_from_wsl.yml
    - Generated from a working & adapted conda env based on Yukuans' environemnt. This can install in WSL without CUDA. Untested on Windows.  
- environment_from_container.yml
    - Generated from the Apptainer image, after installing `environment_from_wsl.yml`. As of writing, this is the exact same file.

### Using conda & environment.yml
#### Exporting an environment
```bash
conda env export --no-builds > environment.yml
```

or, if the environment is not loaded/activated:

```bash
conda env export -p /path/to/conda_env --no-builds > environment.yml
```

#### Loading an environment
```bash
conda env create -f ./mmwave/environment.yml -p ./mmwave/env
```

### Using apptainer
We use apptainer and not Docker, because the DAIC HPC does not (directly) support docker for security reasons. Apptainer does/can use Docker under the hood.

#### Creating a container
The container image has the `*.sif` extension. And can be build as follows:
```bash
apptainer build mmwave_env_container.def
```

This process takes about 10-20 minutes on my laptop, and generates an image of about 8.6GiB. After building the image, you are able to immediately execute it.

#### Using a container

To simply run a container
```bash
apptainer run image.sif
```

To enable NVIDIA libraries
```bash
apptainer run --nv image.sif
```

To bind some directory:
```bash
apptainer run --bind /local/folder:/folder/in/container image.sif
```

As an example, this is the one-liner I use to run the training code on the DAIC HPC:

```bash
apptainer exec --nv --bind /tmp/lucanjannedegr/mwpose3d:/mmwave/mwpose3d /tudelft.net/staff-umbrella/phdvault/mmwave_env_container.sif bash -c "cd /mmwave/mwpose3d && conda run -p ../env python tools/train.py projects/mars/configs/mars_f1p64_b128_e300_mars.py"
```
