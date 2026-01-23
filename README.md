# mwpose3d

## Introduction

mwpose3d is a 3D human pose estimation toolbox built on PyTorch, specifically designed for mmWave radar point cloud data. It provides a modular and extensible framework for developing, training, and evaluating 3D human pose estimation models. The toolbox is built on top of [OpenMMLab's mmengine framework](https://github.com/open-mmlab/mmengine). By leveraging mmengine, mwpose3d offers a flexible configuration system and a standardized training and evaluation pipeline suitable for both research and applied scenarios.

---

## Getting Started

### Installation

mwpose3d can be installed as a Python package following the [installation guide](./docs/INSTALLATION.md).

### Train & Test

The following documents describe the end-to-end workflow for dataset preparation, model training, and evaluation:

- [Public dataset preparation](./docs/DATASET_PREPERATION.md)
- [Custom dataset preparation](./docs/CUSTOM_DATASET.md)
    - [Dataset collection](./docs/CUSTOM_DATASET.md#dataset-collection)
    - [Dataset preparation](./docs/CUSTOM_DATASET.md#dataset-preparation)
- [Gather tracking records](./docs/GENERATE_TRACKING_RECORD.md)
- [Training and evaluating models](./docs/TRAINING_AND_EVALUATING.md)
### Applications

mwpose3d also provides several application-level tools:

- [Custom dataset collection](./docs/CUSTOM_DATASET.md)
- [Online pose estimation](./docs/ONLINE_POSE_ESTIMATION.md)