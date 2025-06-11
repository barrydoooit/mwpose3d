## Generating (Offboard) Tracking Records

For each dataset, we can apply the trackers from mmcore to generate tracked/detected positions of human body, which can be utilized in gaming and pose estimation.

---

### Running the Offline Tracking

   ```bash
   python tools/create_aux_data.py <dataset name> --trec --tcfg <tracker config file path> --pcd-prefix <folder of point cloud data>
   ```

For example, to generate tracking records for the MARS (with outliers) dataset using the GTrack-A tracker, run:
   ```bash
   python tools/create_aux_data.py mars/woutlier --trec --tcfg configs/trackers/gtrack-A.py --pcd-prefix mmwave
   ```
