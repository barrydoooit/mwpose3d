## Custom Dataset

This project supports creating custom datasets using **mmWave radar** synchronized with **Kinect V2** skeletal data. The dataset creation pipeline consists of two stages:

1.  **Dataset Collection** – acquiring raw sensor data
2.  **Dataset Preparation** – aligning, transforming and partitioning the collected data

---

### Dataset Collection

We provide tools to collect synchronized mmWave radar data and Kinect V2 skeleton data.

#### Normal Mode

This mode collects raw sensor data only.

```python
python tools/run_app.py configs.apps/dataset_collection.py
```

#### Running Mode

This mode supports running a human pose estimation (HPE) model simultaneously during data collection.

```python
python tools/run_app.py configs.apps/dataset_collection_and_hpe.py
```

> **Note:** Documentation for generating `DumpKinectSkeleton.exe` is currently under development and will be provided in a future update.


---

### Dataset Preparation

After collecting raw data, the following command is used to create a processed dataset:

```bash
python tools/create_data.py {your_dataset_name} \
    --root-path {path_to_raw_dataset} \
    --out-dir {output_data_directory}
```

A graphical user interface (GUI) will open automatically.

1.  Select the traces you want to include in the dataset.
2.  Click **Align Data** to synchronize radar and Kinect data.
3.  Once the aligned traces appear in the bottom panel, select them and click **Create Data**.

#### Data Partitioning

To split the dataset into different partitions (e.g., train, val, test):

1.  Specify the partition name in the top-right text box.
2.  Select the corresponding traces in the bottom panel.
3.  Click **Allocate to Info** to assign the selected traces to the specified partition.

