# RadHAR: Human Activity Recognition from Point Clouds Generated through a Millimeter-wave Radar

> [RadHAR: Human Activity Recognition from Point Clouds Generated through a Millimeter-wave Radar](https://dl.acm.org/doi/pdf/10.1145/3349624.3356768)

<!-- [ALGORITHM] -->

## Abstract

Accurate human activity recognition (HAR) is the key to enable emerging context-aware applications that require an understanding and identification of human behavior, e.g., monitoring disabled or elderly people who live alone. Traditionally, HAR has been implemented either through ambient sensors, e.g., cameras, or through wearable devices, e.g., a smartwatch, with an inertial measurement unit (IMU). The ambient sensing approach is typically more generalizable for different environments as this does not require every user to have a wearable device. However, utilizing a camera in privacy-sensitive areas such as a home may capture superfluous ambient information that a user may not feel comfortable sharing. Radars have been proposed as an alternative modality for coarse-grained activity recognition that captures a minimal subset of the ambient information using micro-Doppler spectrograms. However, training fine-grained, accurate activity classifiers is a challenge as low-cost millimeter-wave (mmWave) radar systems produce sparse and non-uniform point clouds. In this paper, we propose RadHAR, a framework that performs accurate HAR using sparse and non-uniform point clouds. RadHAR utilizes a sliding time window to accumulate point clouds from a mmWave radar and generate a voxelized representation that acts as input to our classifiers. We evaluate RadHAR using a low-cost, commercial, off-the-shelf radar to get sparse point clouds which are less visually compromising. We evaluate and demonstrate our system on a collected human activity dataset with 5 different activities. We compare the accuracy of various classifiers on the dataset and find that the best performing deep learning classifier achieves an accuracy of 90.47%. Our evaluation shows the efficacy of using mmWave radar for accurate HAR detection and we enumerate future research directions in this space.

## NOTE

> ⚠️ While RadHAR is for HAR (i.e. classification) tasks, we modify the model to be used for pose estimation tasks. 
> 
> ⚠️ To achieve state-of-the-art performance and efficiency, we adopt the Sparse Convolution (spconv) encoder and the SECOND backbone from [this paper](https://www.mdpi.com/1424-8220/18/10/3337?ref=https://codemonkey.link)

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training command

```bash
python tools/train.py projects/radhar/configs/radhar-vcnnblstm_f16_b64_e100_mmfi.py
```

### Testing command

```bash
bash tools/test.py projects/radhar/configs/radhar-vcnnblstm_f16_b64_e100_mmfi.py path/to/checkpoint.pth
```

## Results and model checkpoints

### On MM-Fi Dataset

16 frames model [Model checkpoint](https://drive.google.com/file/d/1fZyf8Za9n5kkvCx-Cfusl8cBCvApcVpp/view?usp=drive_link):

| Joint Type           | MAE (cm) | Joint Type           | MAE (cm) |
|----------------------|---------:|----------------------|---------:|
| 0: Spine Base        |     9.22 | 10: Wrist Right      |    18.33 |
| 1: Spine Mid         |     9.65 | 11: Hand Right       |       – |
| 2: Neck              |    12.87 | 12: Hip Left         |     9.77 |
| 3: Head              |    12.93 | 13: Knee Left        |     9.53 |
| 4: Shoulder Left     |    12.04 | 14: Ankle Left       |     9.78 |
| 5: Elbow Left        |    14.64 | 15: Foot Left        |       – |
| 6: Wrist Left        |    18.45 | 16: Hip Right        |     9.78 |
| 7: Hand Left         |       –  | 17: Knee Right       |     9.14 |
| 8: Shoulder Right    |    11.83 | 18: Ankle Right      |     9.52 |
| 9: Elbow Right       |    14.29 | 19: Foot Right       |       – |


32 frames model [Model checkpoint](https://drive.google.com/file/d/1zLfbcG3hkC_Q2ybGxS51K-LVhIr1flBe/view?usp=drive_link):

| Joint Type           | MAE (cm) | Joint Type           | MAE (cm) |
|----------------------|---------:|----------------------|---------:|
| 0: Spine Base        |     8.90 | 10: Wrist Right      |    18.31 |
| 1: Spine Mid         |     9.26 | 11: Hand Right       |       – |
| 2: Neck              |    12.34 | 12: Hip Left         |     9.38 |
| 3: Head              |    12.40 | 13: Knee Left        |     9.19 |
| 4: Shoulder Left     |    11.80 | 14: Ankle Left       |     9.58 |
| 5: Elbow Left        |    14.63 | 15: Foot Left        |       – |
| 6: Wrist Left        |    17.84 | 16: Hip Right        |     9.54 |
| 7: Hand Left         |       –  | 17: Knee Right       |     9.10 |
| 8: Shoulder Right    |    11.54 | 18: Ankle Right      |     9.76 |
| 9: Elbow Right       |    14.34 | 19: Foot Right       |       – |

