## Raw Data Collection Scripts

This mode collects raw `.bin` radar frames instead of point clouds, utilizing the `UdpRawDataReader` for capturing raw ADC data.

Without GUI (lags less on bad pc)
```python
uv run .\apps\impl\dataset_collection\headless_recorder.py .\configs\apps\raw_dataset_collection.py --duration 1
```

With GUI (lags because of kineect and loses a lot of packets)
```python
python tools/run_app.py configs.apps/raw_dataset_collection.py
```

## Integration with data alignment pipeline

To use the data alignment pipeline, the reader config must be set to process_point_cloud=True when running the data collection script.

```python
reader_cfg = dict(
    type='UdpRawDataReader',
    process_point_cloud=True, # causes much delay
    save_to_file=f'{data_root}/raw/timed_frames.bin'
)
```

```bash
uv run .\tools\create_data.py --root-path "E:\Projects\mwpose3d\apps\impl\dataset_collection\traces\raw_collection" --out-dir "/tmpraw2" custom
```

## Visualizing aligned data

After creating the dataset, can run
```bash
uv run .\tools\inspect_dataset.py projects\rawpose\configs\dca1000evm_default_config.py --vis
```

If you prefer to use process_point_cloud=False, you can use the scripts in `tools/rawproc`: `run_alignment_raw.py` and `visualized_aligned_pc.py`

## TODOs:
[x] fix raw .bin filenames in output dir
[x] integrate with the current alignment pipeline.
[x] separate raw processing -> point cloud dump pipeline to avoid delays in data collection
[] remove tools\rawproc\visualized_aligned_pc.py when not needed anymore
[] finish cleaning code for online collection app apps\impl\dataset_collection\raw_dataset_collection.py
[] script for offline processing of raw data to point clouds (@barrydooit)

## TODOS (26/02)
[] annotate valid pointing gestures in data (using kinect skeleton data)
[] verify point cloud generation matches mmmesh algorithm
[] find precise FOV of radar and calculate minimum range for capturing arm joints
[] calculate precise transformation from kinect and radar to common coordinate system considering both translation and tilt
