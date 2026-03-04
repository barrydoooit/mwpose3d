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
2.  (Optional) click **Calibrate Time** on one selected trace to estimate Kinect/Radar offset.  
    The calibrated value is written into `skeleton_ts_offset_ms` automatically (you can also type this value manually).
3.  Click **Align Data** to synchronize radar and Kinect data using `skeleton_ts_offset_ms`.
4.  Once aligned traces appear in the bottom panel, select them and click **Create Data**.  
    Before creating, set output options in the right panel:
    - `pointcloud_subdir`: subfolder under `mmwave/pointcloud/` (e.g., `default`, `newdsp`).
    - `Structured mmwave_path (dict)`:
      - checked (default): save to `mmwave/pointcloud/<pointcloud_subdir>/`, `mmwave_path` is a dict.
      - unchecked: save to `mmwave/pointcloud/`, `mmwave_path` is a string.

For quick help, hover the `ⓘ` icon next to each field in `Alignment / Output Params`.

> **For Training Config (later step, not data creation):**
> Use `data_prefix` in your model config. Dotted prefix `A.B.C` means path `A/B/C`.
>
> ```python
> data_prefix = dict(
>     pcd='mmwave.pointcloud.default',  # -> mmwave/pointcloud/default
>     skel='skeleton',
> )
> ```

#### Data Partitioning

To split the dataset into different partitions (e.g., train, val, test):

1.  Specify the partition name in the top-right text box.
2.  Select the corresponding traces in the bottom panel.
3.  Click **Allocate to Info** to assign the selected traces to the specified partition.
