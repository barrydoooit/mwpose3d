# mmMesh: Towards 3D Real-Time Dynamic Human Mesh Construction Using Millimeter-Wave

> [mmMesh: Towards 3D Real-Time Dynamic Human Mesh Construction Using Millimeter-Wave](https://dl.acm.org/doi/pdf/10.1145/3458864.3467679)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we present mmMesh, the first real-time 3D humanmesh estimation system using commercial portable millimeterwave devices. mmMesh is built upon a novel deep learning framework that can dynamically locate the moving subject and capturehis/her body shape and pose by analyzing the 3D point cloud generated from the mmWave signals that bounce off the human body. Theproposed deep learning framework addresses a series of challenges.First, it encodes a 3D human body model, which enables mmMeshto estimate complex and realistic-looking 3D human meshes fromsparse point clouds. Second, it can accurately align the 3D pointswith their corresponding body segments despite the influence ofambient points as well as the error-prone nature and the multi-patheffect of the RF signals. Third, the proposed model can infer missing body parts from the information of the previous frames. Ourevaluation results on a commercial mmWave sensing testbed showthat our mmMesh system can accurately localize the vertices onthe human mesh with an average error of 2.47 cm. The superiorexperimental results demonstrate the effectiveness of our proposedhuman mesh construction system.

## Usage

<!-- For a typical model, this section should contain the commands for training and testing. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Training command

```bash
python tools/train.py projects/mmmesh/configs/mmmesh_f32p64_b128_e100_mmfi.py
```

### Testing command

```bash
bash tools/test.py projects/mars/configs/mmmesh_f32p64_b128_e100_mmfi.py path/to/checkpoint.pth
```

## Results and model checkpoints

### On MM-Fi Dataset

[Model checkpoint](https://drive.google.com/file/d/1D5lkFIy-X0UghAgHJhtHgC-HbcOK21Et/view?usp=drive_link):

| Joint Type           | MAE (cm) | Joint Type           | MAE (cm) |
|----------------------|---------:|----------------------|---------:|
| 0: Spine Base        |     9.79 | 10: Wrist Right      |    19.08 |
| 1: Spine Mid         |    10.09 | 11: Hand Right       |       – |
| 2: Neck              |    13.14 | 12: Hip Left         |    10.39 |
| 3: Head              |    13.12 | 13: Knee Left        |    10.11 |
| 4: Shoulder Left     |    12.43 | 14: Ankle Left       |    10.61 |
| 5: Elbow Left        |    15.17 | 15: Foot Left        |       – |
| 6: Wrist Left        |    19.36 | 16: Hip Right        |    10.30 |
| 7: Hand Left         |       – | 17: Knee Right       |     9.72 |
| 8: Shoulder Right    |    12.18 | 18: Ankle Right      |    10.26 |
| 9: Elbow Right       |    14.72 | 19: Foot Right       |       – |



