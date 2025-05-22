# Diffusion Model is a Good Pose Estimator from 3D RF-Vision

> [Diffusion Model is a Good Pose Estimator from 3D RF-Vision](https://arxiv.org/pdf/2403.16198)

<!-- [ALGORITHM] -->

## Abstract

Human pose estimation (HPE) from Radio Frequency vision(RF-vision) performs human sensing using RF signals that penetrateobstacles without revealing privacy (e.g., facial information). Recently,mmWave radar has emerged as a promising RF-vision sensor, providingradar point clouds by processing RF signals. However, the mmWaveradar has a limited resolution with severe noise, leading to inaccurateand inconsistent human pose estimation. This work proposes mmDiff, anovel diffusion-based pose estimator tailored for noisy radar data. Ourapproach aims to provide reliable guidance as conditions to diffusionmodels. Two key challenges are addressed by mmDiff: (1) miss-detectionof parts of human bodies, which is addressed by a module that isolatesfeature extraction from different body parts, and (2) signal inconsistencydue to environmental interference, which is tackled by incorporating priorknowledge of body structure and motion. Several modules are designedto achieve these goals, whose features work as the conditions for thesubsequent diffusion model, eliminating the miss-detection and instabilityof HPE based on RF-vision. Extensive experiments demonstrate thatmmDiff outperforms existing methods significantly, achieving state-ofthe-art performances on public datasets.

## NOTE

> ⚠️ **Warning:** Only the PointTransformer Train/Test is supported by now. The Diffusion module is under examination. For P4Transformer on dense point cloud datasets (e.g., mmBody), please refer to the original repo.

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training command

```bash
python tools/train.py projects/mmdiff/configs/ptransv1_f5p64_b16_e10_mmfi.py
```

### Testing command

```bash
bash tools/test.py projects/mmdiff/configs/ptransv1_f5p64_b16_e10_mmfi.py path/to/checkpoint.pth
```

## Results and model checkpoints

### On MARS Dataset
[Model checkpoint](https://drive.google.com/file/d/1bxvhNmJRq38DGvcE9eJJ4DJAXTonuscw/view?usp=drive_link)

| Joint Type           | MAE (cm) | Joint Type           | MAE (cm) |
|----------------------|---------:|----------------------|---------:|
| 0: Spine Base        |     3.58 | 10: Wrist Right      |     7.59 |
| 1: Spine Mid         |     3.98 | 11: Hand Right       |       – |
| 2: Neck              |     4.52 | 12: Hip Left         |     3.75 |
| 3: Head              |     5.20 | 13: Knee Left        |     3.92 |
| 4: Shoulder Left     |     4.61 | 14: Ankle Left       |     4.29 |
| 5: Elbow Left        |     5.59 | 15: Foot Left        |     4.84 |
| 6: Wrist Left        |     7.44 | 16: Hip Right        |     3.68 |
| 7: Hand Left         |       – | 17: Knee Right       |     3.93 |
| 8: Shoulder Right    |     4.68 | 18: Ankle Right      |     3.87 |
| 9: Elbow Right       |     5.74 | 19: Foot Right       |     4.62 |

### On MM-Fi Dataset

[Model checkpoint](https://drive.google.com/file/d/1EajqxwZWja-UihqQTvukmuPr7R4CSYUi/view?usp=drive_link):

| Joint Type           | MAE (cm) | Joint Type           | MAE (cm) |
|----------------------|---------:|----------------------|---------:|
| 0: Spine Base        |     9.69 | 10: Wrist Right      |    14.88 |
| 1: Spine Mid         |     9.97 | 11: Hand Right       |       – |
| 2: Neck              |    12.93 | 12: Hip Left         |    10.28 |
| 3: Head              |    13.06 | 13: Knee Left        |     9.91 |
| 4: Shoulder Left     |    12.07 | 14: Ankle Left       |    11.17 |
| 5: Elbow Left        |    13.45 | 15: Foot Left        |       – |
| 6: Wrist Left        |    15.42 | 16: Hip Right        |    10.18 |
| 7: Hand Left         |       – | 17: Knee Right       |     9.61 |
| 8: Shoulder Right    |    11.71 | 18: Ankle Right      |    10.69 |
| 9: Elbow Right       |    12.69 | 19: Foot Right       |       – |



