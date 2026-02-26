# generate_pointing_annotation.py
#
# Batch pointing gesture annotator for skeleton H5 files.
# Uses the reusable PointingDetector utility.

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
from tqdm import tqdm

from mmengine.config import Config
from mmengine.registry import DefaultScope

from mwpose3d.utils.pointing_detector import PointingDetector


class PointingGestureAnnotator:
    """
    Walks skeleton H5 files in a dataset and writes per-frame
    boolean ``pointing_gesture`` annotations using PointingDetector.
    """

    H5_DATASET_NAME = "pointing_gesture"

    def __init__(
        self,
        hpe_cfg_f: Union[str, Path],
        *,
        elbow_angle_threshold: float = 160.0,
        stability_window: int = 15,
        stability_max_variance: float = 0.005,
    ) -> None:
        self.hpe_cfg_f = str(hpe_cfg_f)
        self._detector = PointingDetector(
            elbow_angle_threshold=elbow_angle_threshold,
            stability_window=stability_window,
            stability_max_variance=stability_max_variance,
        )

        self._cfg: Optional[Config] = None
        self.data_root: Optional[str] = None

    def generate(self) -> None:
        """Walk skeleton H5 files and annotate each with pointing flags."""
        self._resolve_data_root()
        skel_dir = Path(self.data_root) / "skeleton"
        if not skel_dir.is_dir():
            raise FileNotFoundError(f"Skeleton directory not found: {skel_dir}")

        h5_files = sorted(skel_dir.glob("*.h5"))
        if not h5_files:
            print(f"[pointing] No H5 files found in {skel_dir}")
            return

        for h5_path in tqdm(h5_files, desc="Annotating pointing gestures"):
            self._annotate_file(h5_path)

    def _resolve_data_root(self) -> None:
        """Extract data_root from the HPE config."""
        if self._cfg is None:
            self._cfg = Config.fromfile(self.hpe_cfg_f)
        DefaultScope.get_instance(
            "pointing_annotation", scope_name=self._cfg.default_scope
        )
        for key in ("val_dataloader", "test_dataloader", "train_dataloader"):
            dl_cfg = getattr(self._cfg, key, None)
            if dl_cfg is not None:
                self.data_root = dl_cfg.dataset.data_root
                return
        raise KeyError("No dataloader config found in HPE cfg to extract data_root.")

    def _annotate_file(self, h5_path: Path) -> None:
        """Read skeleton data, compute pointing flags, write back."""
        with h5py.File(h5_path, "r") as f:
            if "skel" not in f:
                return
            skel = f["skel"][:]  # (N_frames, 60)

        flags = self._detector.detect_batch(skel)

        with h5py.File(h5_path, "a") as f:
            if self.H5_DATASET_NAME in f:
                del f[self.H5_DATASET_NAME]
            f.create_dataset(self.H5_DATASET_NAME, data=flags)

        n_pointing = int(flags.sum())
        if n_pointing > 0:
            print(f"  {h5_path.name}: {n_pointing}/{skel.shape[0]} frames flagged as pointing")
