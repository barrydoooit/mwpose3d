# tracking_record_generator_v2.py

from __future__ import annotations
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from tqdm import tqdm

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import DefaultScope


class TrackingRecordGeneratorV2:
    """
    New tracking-centroid generator that consumes the *online* tracker embedded
    in `LoadTrackingRecords` and writes one vlen float32 vector per frame.
    For the *first* sample of each sequence, it drains the whole centroid queue
    (padding with empty arrays on the *front* if needed). For all subsequent
    samples, it takes only the *last* element of the queue.
    """

    def __init__(
        self,
        dataset: str,
        hpe_cfg_f: Union[str, Path],
        *,
        data_prefix: Dict[str, str] = dict(pcd="mmwave"),
        splits: List[str] = ("train", "val", "test"),
        dataloader_key: str = "val_dataloader",
    ) -> None:
        self.dataset = dataset
        self.hpe_cfg_f = str(hpe_cfg_f)
        self.data_prefix = dict(data_prefix, skel="skeleton")
        self.splits = list(splits)
        self.dataloader_key = dataloader_key

        # filled/updated at build time
        self._cfg: Optional[Config] = None
        self._ltr_cfg: Optional[dict] = None  # the LoadTrackingRecords config dict
        self._ltr_tracker_name: Optional[str] = None  # e.g., "RKFTracker"
        self.dataset_root: Optional[str] = None

    def generate(self) -> None:
        """
        Build dataloader from the supplied HPE cfg (trimming pipeline after
        LoadTrackingRecords), iterate samples, and write H5 tracking records.
        """

        for split in self.splits:
            dl = self._make_dataloader(split)
            out_root = Path(self.dataset_root) / "tracking_records"
            out_root.mkdir(parents=True, exist_ok=True)
            self._ltr_tracker_name = self._ltr_cfg.get("tracker_name", "OnlineTracker")
            # storage for current sequence (until we see next starting_flag)
            current_h5_path: Optional[Path] = None
            collected: List[np.ndarray] = []
            collection_size_current_split: List[int] = []

            def flush_sequence() -> None:
                nonlocal current_h5_path, collected
                if current_h5_path is None:
                    collected.clear()
                    return
                with h5py.File(current_h5_path, "a") as h5f:
                    grp_name = self._ltr_tracker_name
                    if grp_name in h5f:
                        del h5f[grp_name]
                    grp = h5f.create_group(grp_name)
                    vlen_dtype = h5py.vlen_dtype(np.dtype("float32"))
                    ds = grp.create_dataset("track_records", shape=(len(collected),), dtype=vlen_dtype)
                    for i, vec in enumerate(collected):
                        ds[i] = vec.astype(np.float32, copy=False)
                    ds.attrs["columns"] = np.array(["x", "y", "z"], dtype="S")
                collected.clear()

            # pcd path key to derive the output file name
            pcd_prefix = self.data_prefix["pcd"]
            pcd_path_key = f"{pcd_prefix}_path"  # e.g., "mmwave_path"

            # Iterate and consume
            iterator = iter(dl)
            total = len(dl) if hasattr(dl, "__len__") else None
            with tqdm(total=total, desc=f"[{split}] Generating tracking records") as pbar:
                while True:
                    try:
                        data = next(iterator)
                    except StopIteration:
                        break
                    # Find file name for this sequence from datadict
                    pcd_files = data.get(pcd_path_key, None)
                    if pcd_files is None:
                        data_file = data.get("data_file", {})
                        pcd_files = data_file.get("pcd", None)
                    if isinstance(pcd_files, (list, tuple)) and len(pcd_files) > 0:
                        pcd_file_name = Path(pcd_files[-1]).name
                    elif isinstance(pcd_files, str):
                        pcd_file_name = Path(pcd_files).name
                    else:
                        pcd_file_name = "unknown.h5"

                    starting_flag: bool = bool(data.get("starting_flag", [False])[0] if isinstance(data.get("starting_flag"), (list, tuple, np.ndarray)) else data.get("starting_flag", False))
                    queue_tb3: List[Tuple[np.ndarray]] = data.get("track_centroid") # [T, B, 3]
                    queue: List[np.ndarray] = [b[-1] for b in queue_tb3]
                    if starting_flag:
                        if len(collected) > 0:
                            collection_size_current_split.append(len(collected))
                        flush_sequence()
                        current_h5_path = out_root / pcd_file_name
                        print(f"Processing sequence: {current_h5_path}")
                        for c in queue:
                            collected.append(c.reshape(-1).astype(np.float32, copy=False))
                    else:
                        # Subsequent frames: only take the last element of the queue
                        last = queue[-1]
                        collected.append(last.reshape(-1).astype(np.float32, copy=False))

                    pbar.update(1)

            collection_size_current_split.append(len(collected))
            assert total + (self.queue_len - 1) * len(collection_size_current_split) == sum(collection_size_current_split), \
                f"Expected {total + (self.queue_len - 1) * len(collection_size_current_split)} frames, but collected {sum(collection_size_current_split)} frames in total."
            # flush last sequence
            flush_sequence()

    @staticmethod
    def _to_vec_or_empty(x: Any) -> np.ndarray:
        """
        Normalize centroid to either empty (shape (0,)) or 3-float vector.
        Accepts None, sequences, numpy arrays, etc.
        """
        if x is None:
            return np.zeros((0,), dtype=np.float32)
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0,), dtype=np.float32)
        arr = arr.reshape(-1)
        if arr.size < 3:
            return np.zeros((0,), dtype=np.float32)
        return arr[:3].astype(np.float32, copy=False)

    def _make_dataloader(self, split: str):
        if self._cfg is None:
            self._cfg = Config.fromfile(self.hpe_cfg_f)
        DefaultScope.get_instance(  # type: ignore
                "generating_tracking_records",
                scope_name=self._cfg.default_scope)
        # Prefer the configured dataloader key (val by default). Fallbacks if needed.
        if not hasattr(self._cfg, self.dataloader_key):
            for key in ("val_dataloader", "test_dataloader", "train_dataloader"):
                if hasattr(self._cfg, key):
                    self.dataloader_key = key
                    break
            else:
                raise KeyError("No usable dataloader config found in the HPE cfg.")

        dl_cfg = deepcopy(getattr(self._cfg, self.dataloader_key))

        ds_cfg = dl_cfg.dataset
        dataset_info_key = "info_path"
        dataset_info_path = ds_cfg.get(dataset_info_key)
        dataset_info_path = str(Path(dataset_info_path).parent / f"info_{split}.pkl")
        self.dataset_root = ds_cfg.data_root
        ds_cfg[dataset_info_key] = dataset_info_path

        pipe = list(ds_cfg.pipeline)
        idx = None
        for i, t in enumerate(pipe):
            if isinstance(t, dict) and t.get("type") == "LoadTrackingRecords":
                idx = i
                break
        if idx is None:
            raise ValueError("No `LoadTrackingRecords` found in the pipeline.")

        # Keep transforms *through* LoadTrackingRecords, drop everything after
        pipe = pipe[: idx + 1]
        ltr_cfg = pipe[-1]
        ltr_cfg["online_mode"] = True
        ltr_cfg.setdefault("anchor_frame", "nearestperframe")

        # Ensure centroid_queue_len is defined — infer from earlier transforms if missing
        self.queue_len = self._infer_queue_len_from_pipeline(pipe)
        ltr_cfg["centroid_queue_len"] = int(self.queue_len)

        # Store for later use (e.g., tracker_name for H5 group)
        self._ltr_cfg = deepcopy(ltr_cfg)
        ds_cfg.pipeline = pipe

        dl_cfg.update(dict(batch_size=1, num_workers=0, persistent_workers=False, shuffle=False))
        # NOTE: Once mmengine sampler is used, we may need to set it
        dl = Runner.build_dataloader(dl_cfg)

        # Patch the *instance* of LoadTrackingRecords in the dataset pipeline to be tolerant at start
        self._patch_ltr_instance(dl.dataset)

        return dl

    @staticmethod
    def _infer_queue_len_from_pipeline(pipeline: List[dict]) -> Optional[int]:
        for t in pipeline:
            if not isinstance(t, dict):
                continue
            typ = t.get("type")
            if typ == "LoadMultiFrameFromH5" and "num_frames" in t:
                return int(t["num_frames"])
        return None

    def _patch_ltr_instance(self, dataset) -> None:
        """
        Patch both the *instance* and the *class* of LoadTrackingRecords so
        'queue not fulfilled' never raises during visualization/generation.
        """
        transforms = getattr(dataset, "pipeline", None)
        if transforms is None:
            return

        pipeline_objs = getattr(transforms, "transforms", [])
        for t in pipeline_objs:
            if t.__class__.__name__ == "LoadTrackingRecords":
                if hasattr(t, "tracker_name"):
                    self._ltr_tracker_name = getattr(t, "tracker_name")
                t.online_noresult_response = 'empty'
                break

    def visualize(
        self,
        fps: int = 30,
        tracking_mode: str = "dot",
    ) -> None:
        """
        Visualize the tracking centroids produced by LoadTrackingRecords (online mode).
        For each dataloader sample we:
          - show the last PCD frame,
          - show the last skeleton frame (if available),
          - show the centroid for the *current* frame (last element of the queue),
            or an empty array if missing.

        Notes:
        - We do NOT expand the initial queue into multiple frames to keep parity with
          the original visualizer behavior (one visual frame per dataloader step).
        - Dataloader is single-worker for deterministic, stateful tracking.
        """
        split = self.splits[0]
        dl = self._make_dataloader(split)
        total = len(dl) if hasattr(dl, "__len__") else None

        try:
            from PySide6.QtWidgets import QApplication
        except Exception:
            QApplication = None

        # Import your visualizer (adjust import path to your project)
        try:
            from mwpose3d.visualization import PointCloudOfflineVisualizerSK
        except Exception:
            try:
                from visualization import PointCloudOfflineVisualizerSK
            except Exception as e:
                raise ImportError(
                    "Cannot import PointCloudOfflineVisualizerSK; adjust the import path."
                ) from e

        def _safe_iter(dataloader):
            """Yield data dicts, but never raise: on error, yield a placeholder item."""
            it = iter(dataloader)
            while True:
                try:
                    yield next(it)
                except StopIteration:
                    return
                except RuntimeError as e:
                    # Minimal placeholder with no overlays
                    yield {"pcd_frames": None, "skel_frames": None, "track_centroid": ()}

        # Each generator independently iterates through the same dataloader sequence
        # (the visualizer advances them in lockstep).
        def pcd_generator():
            for data in _safe_iter(dl):
                pcd_frames = data.get("pcd_frames", None)
                if pcd_frames is None:
                    yield None
                    continue
                try:
                    yield pcd_frames[-1][0]
                except Exception:
                    # Try a few fallbacks
                    try:
                        yield pcd_frames[-1]
                    except Exception:
                        yield None

        def skel_generator():
            for data in _safe_iter(dl):
                skel_frames = data.get("skel_frames", None)
                if skel_frames is None:
                    yield None
                    continue
                try:
                    yield skel_frames[-1][0]
                except Exception:
                    try:
                        yield skel_frames[-1]
                    except Exception:
                        yield None

        def trk_generator():
            for data in _safe_iter(dl):
                tc = data.get("track_centroid", None)
                last = None
                if tc is None:
                    raise ValueError("No track_centroid in datadict")
                while isinstance(tc, (list, tuple)):
                    tc = tc[-1]
                print(tc)
                if tc.size < 3:
                    yield np.zeros((0, 3), dtype=np.float32)
                else:
                    yield tc.reshape(1, 3)

        app = None
        if QApplication is not None:
            app = QApplication.instance() or QApplication(sys.argv)

        visualizer = PointCloudOfflineVisualizerSK(
            point_clouds=pcd_generator(),
            skeletons=skel_generator(),
            tracking_data=trk_generator(),
            total_frames=total,
            play_fps=fps,
            tracking_mode=tracking_mode,
        )
        visualizer.show()
        if app is not None:
            app.exec_()

# ---------------------------
# If you want a minimal CLI:
# ---------------------------
# if __name__ == "__main__":
#     gen = TrackingRecordGeneratorV2(
#         dataset="YOUR_DATASET_NAME",
#         tracker_cfg_f="path/to/tracker_cfg.py",
#         hpe_cfg_f="path/to/hpe_cfg.py",
#         splits=["train", "val", "test"],   # adjust as needed
#     )
#     gen.generate()
