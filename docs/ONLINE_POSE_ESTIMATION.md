## Online Pose Estimation Application

We support running trained 3D human pose estimation (HPE) models in an online inference mode, enabling real-time pose estimation from live mmWave radar data. This application processes streaming radar inputs and visualizes predicted human poses with minimal latency, making it suitable for real-time demonstrations and system validation.

### Example Usage

```bash
python tools/run_app.py configs.apps/read_hpe_vis.py
```

This command launches the online inference pipeline, which:
- Continuously reads live mmWave radar data
- Performs real-time 3D pose estimation using a trained model
- Visualizes the predicted skeletal poses