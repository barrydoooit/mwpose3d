"""
PointCloudReprocessor: re-processes raw ADC .bin files with an alternative DSP
config to generate a new point cloud variant in an existing formatted dataset.

The formatted dataset must already have a 'default' point cloud variant (produced
by the standard create_data.py pipeline).  The new variant is saved under:
    <dataset_dir>/mmwave/pointcloud/<variant_name>/<episode_id>.h5

and every info_*.pkl file that contains the episode is updated so that
mmwave_path['pointcloud'][<variant_name>] points to the new file.

Alignment note
--------------
alignment.py adjusts skeleton timestamps to match PCD timestamps, leaving the PCD
timestamps unchanged.  The 'ts' values stored in the default HDF5 therefore equal
the timestamps embedded in the raw .bin file (8-byte float64 ms).  Matching is
done by rounding the float64 to the nearest integer ms.
"""

import pickle
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


class PointCloudReprocessor:
    """
    Re-process raw ADC .bin files for each episode in a formatted dataset using
    a new DSP config and write the results as an additional point cloud variant.

    Parameters
    ----------
    formatted_dataset_dir : Path
        Root of the formatted dataset (contains info_all.pkl, mmwave/, skeleton/).
    raw_traces_dir : Path
        Root of the raw trace collection (contains raw/<episode>.bin, etc.).
    dsp_cfg_path : Path
        Path to a mwpose3d DSP config .py file (defines mmwave_radar_cfg,
        dsp_pipeline_cfg, and has_timestamp).
    variant_name : str, optional
        Sub-directory name used under mmwave/pointcloud/.  Defaults to the
        config filename stem (e.g. "ti-mobile-tracker-64loops_xWR1843").
    """

    def __init__(self,
                 formatted_dataset_dir: Path,
                 raw_traces_dir: Path,
                 dsp_cfg_path: Path,
                 variant_name: Optional[str] = None):
        self.formatted_dataset_dir = Path(formatted_dataset_dir)
        self.raw_traces_dir = Path(raw_traces_dir)
        self.dsp_cfg_path = Path(dsp_cfg_path)
        self.variant_name = variant_name or self.dsp_cfg_path.stem

        self._pipeline = None
        self._has_timestamp: Optional[bool] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_all(self, episode_ids: Optional[List[str]] = None):
        """Process all (or a specified subset of) episodes."""
        info_path = self.formatted_dataset_dir / 'info_all.pkl'
        if not info_path.exists():
            raise FileNotFoundError(f'info_all.pkl not found in {self.formatted_dataset_dir}')

        with open(info_path, 'rb') as f:
            info_all = pickle.load(f)

        target_ids = set(episode_ids) if episode_ids else None
        for entry in info_all:
            episode_id = entry['id']
            if target_ids and episode_id not in target_ids:
                continue
            print(f'[PointCloudReprocessor] Processing episode: {episode_id}')
            try:
                self.process_episode(episode_id)
            except Exception:
                print(f'[PointCloudReprocessor] Error processing {episode_id}:')
                traceback.print_exc()

    def process_episode(self, episode_id: str):
        """Re-process a single episode and update the info files."""
        frame_list = self._load_frame_list(episode_id)
        ts_to_pcd = self._process_bin_file(episode_id, frame_list)
        out_path = self._write_episode_h5(episode_id, frame_list, ts_to_pcd)
        self._update_all_info_files(episode_id)
        matched = sum(1 for _, ts in frame_list if ts is not None and ts in ts_to_pcd)
        print(
            f'[PointCloudReprocessor] Saved {out_path.name}: '
            f'{matched}/{len(frame_list)} frames matched'
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pipeline(self):
        """Lazy-load the DSP pipeline from the config file."""
        if self._pipeline is not None:
            return self._pipeline, self._has_timestamp

        from mmengine.config import Config
        from mwcore.signal_processing.dsp_ppl import DspPipeline
        from mwcore.signal_processing.frame import RadarConfig

        cfg = Config.fromfile(str(self.dsp_cfg_path))

        has_timestamp = bool(cfg.get('has_timestamp', False))
        if not has_timestamp:
            raise ValueError(
                f"DSP config '{self.dsp_cfg_path.name}' has has_timestamp=False (or unset). "
                "has_timestamp=True is required so the reader knows each frame has an 8-byte "
                "timestamp header; without it frame boundaries are misaligned."
            )

        radar_cfg = RadarConfig(**cfg.mmwave_radar_cfg)
        pipeline = DspPipeline(radar_cfg, list(cfg.dsp_pipeline_cfg))

        self._pipeline = pipeline
        self._has_timestamp = has_timestamp
        return self._pipeline, self._has_timestamp

    def _load_frame_list(self, episode_id: str) -> List[Tuple[int, int]]:
        """
        Load the per-frame (seq, ts) list from the default HDF5.

        Returns a list ordered by frame position.  ts may be None for empty
        frames (rare; produced by the 'remove' missing-frame strategy).
        """
        default_h5 = (
            self.formatted_dataset_dir / 'mmwave' / 'pointcloud' /
            'default' / f'{episode_id}.h5'
        )
        if not default_h5.exists():
            raise FileNotFoundError(
                f"Default HDF5 not found: {default_h5}\n"
                "Run create_data.py first to generate the default point cloud variant."
            )

        with h5py.File(default_h5, 'r') as f:
            data = f['pcd/data'][:]
            index = f['pcd/index'][:]
            columns = [c.decode('utf-8') for c in f['pcd/data'].attrs['columns']]

        ts_col = columns.index('ts')
        seq_col = columns.index('seq')

        frame_list: List[Tuple[int, int]] = []
        for i in range(len(index) - 1):
            start, end = int(index[i]), int(index[i + 1])
            if start < end:
                ts = int(data[start, ts_col])
                seq = int(data[start, seq_col])
            else:
                # Empty frame: assign seq by position, ts unknown
                ts = None
                seq = i
            frame_list.append((seq, ts))

        return frame_list

    def _process_bin_file(self,
                           episode_id: str,
                           frame_list: List[Tuple[int, int]]) -> Dict[int, np.ndarray]:
        """
        Read the raw .bin file and return a {ts: point_cloud(N,5)} mapping.

        point_cloud columns: [x, y, z, vel, snr]  (range column dropped to
        match the 5-feature format used in the existing HDF5 files).

        Timestamp matching uses int() truncation to mirror load_utils.py
        which stores int(ts) in the HDF5.  The raw .bin timestamps are
        float64 ms with sub-ms fractional parts; int() gives the same
        integer-ms value that the JSON/HDF5 pipeline produced.
        """
        from mwcore.radario.readers.offlineReaders.adcbin_reader import OfflineAdcDataReader

        pipeline, has_timestamp = self._get_pipeline()
        bin_dir = self.raw_traces_dir / 'raw'

        if not bin_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {bin_dir}")

        bin_file = bin_dir / f'{episode_id}.bin'
        if not bin_file.exists():
            raise FileNotFoundError(f"Raw .bin file not found: {bin_file}")

        # Restrict the reader to only this episode's file
        file_pattern = f'^{re.escape(episode_id)}\\.bin$'

        reader = OfflineAdcDataReader(
            data_dir=str(bin_dir),
            file_pattern=file_pattern,
            has_timestamp=has_timestamp,
            pipeline=pipeline,
        )

        expected_ts = {ts for _, ts in frame_list if ts is not None}
        ts_to_pcd: Dict[int, np.ndarray] = {}

        for frame in reader:
            raw_ts = frame.frame_start_timestamp_ms
            if raw_ts is None:
                continue
            ts = int(raw_ts)
            if ts not in expected_ts:
                continue
            # point_cloud shape: (N, 6) — [x, y, z, vel, snr, range]
            pc = frame.point_cloud
            # Drop range column to match the 5-feature format of existing HDF5 files
            ts_to_pcd[ts] = pc[:, :5].astype(np.float64)

        reader.close()
        return ts_to_pcd

    def _write_episode_h5(self,
                           episode_id: str,
                           frame_list: List[Tuple[int, int]],
                           ts_to_pcd: Dict[int, np.ndarray]) -> Path:
        """Write the new variant HDF5 file and return its path."""
        out_dir = (
            self.formatted_dataset_dir / 'mmwave' / 'pointcloud' / self.variant_name
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{episode_id}.h5'

        columns = ['x', 'y', 'z', 'vel', 'snr', 'seq', 'ts']

        all_points: List[np.ndarray] = []
        frame_indices: List[int] = [0]

        for seq, ts in frame_list:
            if ts is not None and ts in ts_to_pcd:
                pc = ts_to_pcd[ts]           # (N, 5)
                n = len(pc)
                seq_col = np.full((n, 1), seq, dtype=np.float64)
                ts_col = np.full((n, 1), ts, dtype=np.float64)
                frame_data = np.hstack([pc, seq_col, ts_col])
            else:
                frame_data = np.zeros((0, 7), dtype=np.float64)

            all_points.append(frame_data)
            frame_indices.append(frame_indices[-1] + len(frame_data))

        non_empty = [a for a in all_points if len(a) > 0]
        if non_empty:
            pcd_all = np.concatenate(non_empty, axis=0)
        else:
            pcd_all = np.zeros((0, 7), dtype=np.float64)

        frame_indices_arr = np.array(frame_indices, dtype=np.int64)

        with h5py.File(out_path, 'w') as f:
            grp = f.create_group('pcd')
            ds = grp.create_dataset('data', data=pcd_all)
            grp.create_dataset('index', data=frame_indices_arr)
            ds.attrs['columns'] = np.array(columns, dtype='S')

        return out_path

    def _update_all_info_files(self, episode_id: str):
        """Update every info_*.pkl in the dataset dir that contains this episode."""
        rel_path = f'pointcloud/{self.variant_name}/{episode_id}.h5'

        for info_pkl in sorted(self.formatted_dataset_dir.glob('info_*.pkl')):
            with open(info_pkl, 'rb') as f:
                info_all = pickle.load(f)

            updated = False
            for entry in info_all:
                if entry['id'] != episode_id:
                    continue

                mmwave = entry.get('mmwave_path', entry.get('mmwave', {}))
                pointcloud_map = self._extract_pointcloud_map(mmwave)
                pointcloud_map[self.variant_name] = rel_path

                entry['mmwave_path'] = {'pointcloud': pointcloud_map}
                entry.pop('mmwave', None)
                updated = True
                break

            if updated:
                with open(info_pkl, 'wb') as f:
                    pickle.dump(info_all, f)

    @staticmethod
    def _extract_pointcloud_map(mmwave_value) -> dict:
        """Extract the pointcloud variant dict from an mmwave_path value."""
        if isinstance(mmwave_value, str):
            parts = Path(mmwave_value).parts
            if 'pointcloud' in parts:
                i = parts.index('pointcloud')
                if len(parts) >= i + 2:
                    variant = parts[i + 1]
                    return {variant: mmwave_value}
            return {'default': mmwave_value}
        if isinstance(mmwave_value, dict):
            pointcloud = mmwave_value.get('pointcloud', {})
            if isinstance(pointcloud, dict):
                return dict(pointcloud)
        return {}
