
from copy import deepcopy
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import pickle
import threading
from abc import ABC, abstractmethod
import traceback
from typing import Dict, Optional, Protocol, List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from .widgets import CheckList, HoverTooltip, MultiColumnCheckList
from ...rawproc.episode import Episode
from ...rawproc.hdf5_dumper import ToHdf5


class DataProcessorProtocol(Protocol):
    @property
    @abstractmethod
    def processed_episode_names(self) -> set: ...
    @abstractmethod
    def load_processed_episodes(self) -> list: ...

    @property
    def episodes(self) -> Dict[str, Episode]: ...
    @abstractmethod
    def get_episode(self, name: str) -> Episode: ...
    @abstractmethod
    def update_episode(self, name: str, new_episode: Episode): ...

    @abstractmethod
    def handle_create_data(
        self,
        episodes: list,
        suffix: str,
        pointcloud_subdir: str,
        mmwave_path_as_dict: bool,
    ): ...
    @abstractmethod
    def handle_delete_raw(self, episodes: list): ...
    @abstractmethod
    def handle_allocate_to_info(self, episodes: list, suffix: str): ...
    @abstractmethod
    def handle_calibrate_time(self, episodes: list): ...
    @abstractmethod
    def handle_remove_processed(self, episodes: list, also_data: bool): ...
    @abstractmethod
    def handle_reload_raw(self, episodes: list): ...
    @abstractmethod
    def handle_data_align(self, episodes: list, skeleton_ts_offset_ms: int): ...


class DataProcessorGUI(tk.Tk):
    """
    Reimplemented GUI wiring to support:
    - Shift-click multi-select in both episode lists.
    - Mouse wheel scrolling on hover.
    - Hide 'Calibrated' in both lists, and hide 'Aligned' in the bottom (Processed) list.
    - Display per-episode _pcd_meta key/value pairs as columns in both lists (fill '-' for missing).
    - Clickable headers to sort by any column (toggle on repeat; episode name is tie-breaker ascending).
    """
    def __init__(self, processor: DataProcessorProtocol):
        super().__init__()
        self.title("Data Processing GUI")
        self.geometry("1000x700")
        self.processor = processor

        self._buttons: Dict[str, ttk.Button] = {}
        self._tooltips: List[HoverTooltip] = []

        self.create_widgets()
        self.processor.load_processed_episodes()
        self.refresh_episode_lists()
        self.processor._gui_refresh_callabck = self.refresh_episode_lists
        self.processor._gui_set_offset_callback = self.set_skeleton_ts_offset

    def create_widgets(self):
        # Two-column layout
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left: episode lists
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # We will provide columns dynamically in set_items (so pass no columns here)
        self.unprocessed_list = MultiColumnCheckList(
            list_frame, "Unprocessed Episodes", self.toggle_buttons_state
        )
        self.unprocessed_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.processed_list = MultiColumnCheckList(
            list_frame, "Processed Episodes", self.toggle_buttons_state
        )
        self.processed_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right: control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(control_frame, text="Info Suffix:").pack(anchor=tk.W)
        self.info_suffix_entry = ttk.Entry(control_frame)
        self.info_suffix_entry.pack(fill=tk.X, pady=5)

        param_frame = ttk.LabelFrame(control_frame, text="Alignment / Output Params")
        param_frame.pack(fill=tk.X, pady=5)

        offset_row = ttk.Frame(param_frame)
        offset_row.pack(fill=tk.X)
        ttk.Label(offset_row, text="skeleton_ts_offset_ms:").pack(side=tk.LEFT)
        offset_info = ttk.Label(offset_row, text="ⓘ", foreground="#1f6aa5", cursor="question_arrow")
        offset_info.pack(side=tk.LEFT, padx=(4, 0))
        self._tooltips.append(HoverTooltip(
            offset_info,
            "Time shift (ms) to be applied when clicking Align Data. "
            "Use Calibrate Time to estimate this value, or type it manually."
            "+X means skeleton frame arrives X ms before radar frame;"
            "-X means skeleton frame arrives X ms after radar frame."
        ))
        self.skeleton_offset_entry = ttk.Entry(param_frame)
        self.skeleton_offset_entry.insert(0, "0")
        self.skeleton_offset_entry.pack(fill=tk.X, pady=2)

        subdir_row = ttk.Frame(param_frame)
        subdir_row.pack(fill=tk.X)
        ttk.Label(subdir_row, text="pointcloud_subdir:").pack(side=tk.LEFT)
        subdir_info = ttk.Label(subdir_row, text="ⓘ", foreground="#1f6aa5", cursor="question_arrow")
        subdir_info.pack(side=tk.LEFT, padx=(4, 0))
        self._tooltips.append(HoverTooltip(
            subdir_info,
            "Subfolder under mmwave/pointcloud/ for created H5 files "
            "(for example: default, newdsp)."
        ))
        self.pointcloud_subdir_entry = ttk.Entry(param_frame)
        self.pointcloud_subdir_entry.insert(0, "default")
        self.pointcloud_subdir_entry.pack(fill=tk.X, pady=2)

        structured_row = ttk.Frame(param_frame)
        structured_row.pack(fill=tk.X)
        self.mmwave_path_mode_var = tk.BooleanVar(value=True)
        self.mmwave_path_mode_check = ttk.Checkbutton(
            structured_row,
            text="Structured mmwave_path (dict)",
            variable=self.mmwave_path_mode_var,
            command=self._on_mmwave_path_mode_changed,
        )
        self.mmwave_path_mode_check.pack(side=tk.LEFT, pady=2)
        structured_info = ttk.Label(structured_row, text="ⓘ", foreground="#1f6aa5", cursor="question_arrow")
        structured_info.pack(side=tk.LEFT, padx=(4, 0))
        self._tooltips.append(HoverTooltip(
            structured_info,
            "Checked: save to mmwave/pointcloud/<pointcloud_subdir>/ and store mmwave_path as dict.\n"
            "Unchecked: save to mmwave/pointcloud/ and store mmwave_path as string."
        ))
        self._on_mmwave_path_mode_changed()

        self.create_control_buttons(control_frame)

        # Status bar
        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var).pack(side=tk.BOTTOM, fill=tk.X)

    def create_control_buttons(self, master=None):
        button_configs = [
            ("Create Data", self.create_data),
            ("Delete Raw", self.delete_raw),
            ("Allocate to Info", self.allocate_to_info),
            ("Calibrate Time", self.calibrate_time),
            ("Remove Processed", self.remove_processed),
            ("Reload Raw", self.reload_raw),
            ("Align Data", self.align_data),
        ]
        for text, cmd in button_configs:
            self.add_custom_button(master, text, cmd)

    def add_custom_button(self, master: Optional[ttk.Frame], text: str, handler: callable):
        new_btn = ttk.Button(self if master is None else master, text=text, command=handler)
        new_btn.pack(fill=tk.X, pady=5)
        self._buttons[text] = new_btn

    def get_selected_episodes(self, lazy=True):
        if lazy:
            return tuple(dict.fromkeys(self.unprocessed_list.checked_items + self.processed_list.checked_items))

        return {
            name: self.processor.get_episode(name)
            for name in self.unprocessed_list.checked_items + self.processed_list.checked_items
        }

    def toggle_buttons_state(self, _=None):
        has_selection = bool(self.unprocessed_list.checked_items or self.processed_list.checked_items)
        state = tk.NORMAL if has_selection else tk.DISABLED
        for btn in self._buttons.values():
            btn.config(state=state)

    # ---- Event handles delegated to processor ----
    def create_data(self):
        episodes = self.get_selected_episodes(lazy=False)
        suffix = self.info_suffix_entry.get().strip()
        mmwave_path_as_dict = bool(self.mmwave_path_mode_var.get())
        if mmwave_path_as_dict:
            pointcloud_subdir = self.pointcloud_subdir_entry.get().strip() or "default"
        else:
            # Flat output mode: write to mmwave/pointcloud directly.
            pointcloud_subdir = ""
        self.processor.handle_create_data(
            episodes,
            suffix,
            pointcloud_subdir,
            mmwave_path_as_dict=mmwave_path_as_dict,
        )

    def delete_raw(self):
        episode_names = self.get_selected_episodes(lazy=True)
        if len(episode_names) > 0:
            if messagebox.askyesno("Delete Raw Data", "Are you sure to delete raw data?"):
                self.processor.handle_delete_raw(episode_names)

    def allocate_to_info(self):
        episode_names = self.get_selected_episodes(lazy=True)
        suffix = self.info_suffix_entry.get().strip()
        self.processor.handle_allocate_to_info(episode_names, suffix)

    def calibrate_time(self):
        episode_names = self.get_selected_episodes(lazy=True)
        if len(episode_names) > 0:
            self.processor.handle_calibrate_time(episode_names)

    def remove_processed(self):
        episode_names = self.get_selected_episodes(lazy=True)
        also_data = messagebox.askyesno("Remove Processed Data", "Also remove processed data?")
        self.processor.handle_remove_processed(episode_names, also_data)

    def reload_raw(self):
        episode_names = self.get_selected_episodes(lazy=True)
        self.processor.handle_reload_raw(episode_names)

    def align_data(self):
        episode_names = self.get_selected_episodes(lazy=True)
        try:
            skeleton_ts_offset_ms = int(self.skeleton_offset_entry.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Parameter", "skeleton_ts_offset_ms must be an integer.")
            return
        self.processor.handle_data_align(episode_names, skeleton_ts_offset_ms=skeleton_ts_offset_ms)

    def set_skeleton_ts_offset(self, offset_ms: int):
        self.skeleton_offset_entry.delete(0, tk.END)
        self.skeleton_offset_entry.insert(0, str(int(offset_ms)))

    def _on_mmwave_path_mode_changed(self):
        if self.mmwave_path_mode_var.get():
            self.pointcloud_subdir_entry.config(state=tk.NORMAL)
        else:
            self.pointcloud_subdir_entry.config(state=tk.DISABLED)

    # ---- ----

    def update_ui_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in self._buttons.values():
            btn.config(state=state)

    def _collect_meta(self, episode_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Return (meta_df, sorted_meta_keys) for provided episode names.

        meta_df index is episode, columns are meta keys, values are strings or '-' for missing.

        We read only _pcd_meta (fast) instead of loading full episode data.

        """
        meta_dir = self.processor.raw_dir / "meta"
        meta_records = {}
        all_keys = set()
        for ep in episode_names:
            meta = {}
            try:
                # Only load meta to avoid heavy loads.
                ep_obj = Episode(ep)
                ep_obj.load_pcd_meta(meta_dir)
                if hasattr(ep_obj, "_pcd_meta") and isinstance(ep_obj._pcd_meta, dict):
                    meta = ep_obj._pcd_meta
            except Exception as e:
                logger.warning(f"Failed to load meta for {ep}: {e}")
            meta_records[ep] = meta
            all_keys.update(meta.keys())

        meta_df = pd.DataFrame.from_dict(meta_records, orient="index")
        if len(all_keys) > 0:
            meta_df = meta_df.reindex(columns=sorted(all_keys))
        meta_df = meta_df.fillna("-")
        return meta_df, sorted(all_keys)

    def refresh_episode_lists(self):
        # Determine available episodes from raw pointclouds
        radar_dir = self.processor.raw_dir / "pointcloud"
        if radar_dir.exists():
            radar_files = list(radar_dir.glob("*"))
            episode_names = {f.stem for f in radar_files if not f.is_dir()}
        else:
            episode_names = set()

        # Ensure status rows exist
        for ep in episode_names:
            if ep not in self.processor.status_df.index:
                self.processor.status_df.loc[ep] = [False] * len(self.processor.status_df.columns)

        processed = sorted(self.processor.processed_episode_names)
        unprocessed = sorted(episode_names - self.processor.processed_episode_names)

        # Build meta columns for all episodes
        all_eps_sorted = sorted(list(episode_names))
        meta_df_all, meta_keys_sorted = self._collect_meta(all_eps_sorted)

        # Build display dataframes by merging status (ops) + meta
        # NOTE: We hide 'Calibrated' everywhere and hide 'Aligned' for processed list.
        # Keep booleans in the df; missing filled downstream in widget.
        status_all = self.processor.status_df.reindex(all_eps_sorted).copy()
        for col in ["Calibrated", "Aligned"]:
            if col not in status_all.columns:
                status_all[col] = False

        display_all = pd.concat([status_all, meta_df_all], axis=1)

        # Slice to subsets
        unprocessed_df = display_all.reindex(unprocessed)
        processed_df = display_all.reindex(processed)

        # Columns to show
        # Hide Calibrated in both; show Aligned only on unprocessed; meta columns appear on both.
        unprocessed_cols: List[str] = list(meta_keys_sorted) + (["Aligned"] if "Aligned" in display_all.columns else [])
        processed_cols: List[str] = list(meta_keys_sorted)  # no Calibrated, no Aligned

        # Send to widgets
        self.unprocessed_list.set_items(unprocessed_df, columns=unprocessed_cols)
        self.processed_list.set_items(processed_df, columns=processed_cols)

    def show_status(self, message: str, error: bool = False):
        fg = "red" if error else "black"
        self.status_var.set(message)
        self.after(10000, lambda: self.status_var.set(""))


class DataProcessorDelegate(DataProcessorProtocol):
    def __init__(self, raw_dir: str, output_dir):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self._processed_episode_names = set()
        self._episodes: Dict[str, Episode] = {}
        self._gui_refresh_callabck = None
        self._gui_set_offset_callback = None
        self._status_df = pd.DataFrame(columns=["Calibrated", "Aligned"], dtype=bool)

    @property
    def status_df(self):
        return self._status_df

    @property
    def episodes(self):
        return self._episodes

    @property
    def processed_episode_names(self):
        return self._processed_episode_names

    def load_processed_episodes(self):
        info_path = self.output_dir / "info_all.pkl"
        if info_path.exists():
            with open(info_path, "rb") as f:
                self._processed_episode_names = {info["id"] for info in pickle.load(f)}

    def get_episode(self, name: str):
        if name in self.episodes:
            return self.episodes[name]
        episode = Episode(name)
        episode.load_pcd(self.raw_dir / "pointcloud")
        episode.load_pcd_meta(self.raw_dir / "meta")
        episode.load_skeleton(self.raw_dir / "kinect")
        self.episodes[name] = episode
        return episode

    def update_episode(self, name, new_episode):
        assert isinstance(new_episode, Episode)
        self.episodes[name] = new_episode
        logger.info(f"Episode {name} updated.")

    def gui_refresh_callback(self):
        if self._gui_refresh_callabck:
            self._gui_refresh_callabck()

    def handle_create_data(
        self,
        episodes: list,
        suffix: str,
        pointcloud_subdir: str,
        mmwave_path_as_dict: bool,
    ):
        def task():
            suffixes = ["all"]
            if suffix:
                suffixes.append(suffix)
            for name in episodes:
                try:
                    episode = self.get_episode(name)
                    hdf5_maker = ToHdf5(
                        episode,
                        self.output_dir,
                        pointcloud_subdir=pointcloud_subdir,
                        mmwave_path_as_dict=mmwave_path_as_dict,
                    )
                    hdf5_maker.save(suffixes)
                    self._processed_episode_names.add(name)
                except Exception as e:
                    logger.error(f"Failed to create data: {e}")
            self.gui_refresh_callback()

        threading.Thread(target=task, daemon=True).start()

    def handle_delete_raw(self, episodes: list):
        def task():
            for name in episodes:
                try:
                    pointcloud_candidates = [self.raw_dir / "pointcloud" / f"{name}.json"]
                    pointcloud_candidates += list((self.raw_dir / "pointcloud").glob(f"*/{name}.json"))
                    paths = pointcloud_candidates + [
                        self.raw_dir / "meta" / f"{name}.json",
                        self.raw_dir / "meta" / f"{name}.meta.json",
                        self.raw_dir / "kinect" / f"{name}.csv",
                    ]
                    for p in paths:
                        if p.exists():
                            p.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete raw data: {e}")
            self.gui_refresh_callback()

        threading.Thread(target=task, daemon=True).start()

    def handle_allocate_to_info(self, episodes: list, suffix: str):
        def task():
            try:
                info_all = []
                info_pkl_path = self.output_dir / "info_all.pkl"
                if info_pkl_path.exists():
                    with open(info_pkl_path, "rb") as f:
                        info_all = pickle.load(f)
                new_info = []
                new_info_path = self.output_dir / f"info_{suffix}.pkl"
                if new_info_path.exists():
                    with open(new_info_path, "rb") as f:
                        new_info = pickle.load(f)
                for name in episodes:
                    try:
                        key = name
                        _info = None
                        for infoa in info_all:
                            if infoa["id"] == key:
                                _info = deepcopy(infoa)
                                break
                        else:
                            raise KeyError(f"Episode {name} not found in info_all. Create data first.")
                        for infon in new_info:
                            infon: dict
                            if infon["id"] == key:
                                logger.warning(
                                    f"Episode {name} already exists in info_{suffix}.pkl. Coverwriting."
                                )
                                infon.clear()
                                infon.update(_info)
                        else:
                            new_info.append(deepcopy(_info))
                            logger.info(f"Episode {name} added to info_{suffix}.pkl.")

                    except Exception as e:
                        logger.error(f"Failed to allocate to info: {e}")
                with open(new_info_path, "wb") as f:
                    pickle.dump(new_info, f)
            except Exception as e:
                logger.error(f"Failed to allocate to info: {e}")

        threading.Thread(target=task, daemon=True).start()

    def handle_calibrate_time(self, episodes: list):
        if not episodes:
            return
        # Calibrate only on the first selected episode and return offset only.
        name = episodes[0]
        try:
            episode = self.get_episode(name)
            offset_ms = episode.calibrate_time(manual=True)
            if offset_ms is None:
                return
            self.status_df.loc[name, "Calibrated"] = True
            if self._gui_set_offset_callback is not None:
                self._gui_set_offset_callback(int(offset_ms))
            logger.info("Manual calibration finished on %s. skeleton_ts_offset_ms=%s", name, offset_ms)
            self.gui_refresh_callback()
        except Exception as e:
            logger.error(f"Failed to calibrate time: {e}")
            traceback.print_exc()

    def handle_remove_processed(self, episodes, also_data):
        def task():
            try:
                all_info_files = list(self.output_dir.glob("info_*.pkl"))
                for name in episodes:
                    if also_data:
                        paths = [self.output_dir / "h5" / f"{name}.h5"]
                        for p in paths:
                            if p.exists():
                                p.unlink()
                    for info_file in all_info_files:
                        with open(info_file, "rb") as f:
                            info = pickle.load(f)
                        if f"{name}.h5" in info:
                            info.pop(f"{name}.h5")
                        with open(info_file, "wb") as f:
                            pickle.dump(info, f)
                    self._processed_episode_names.discard(name)
                self.gui_refresh_callback()
            except Exception as e:
                logger.error(f"Failed to remove processed data: {e}")

        threading.Thread(target=task, daemon=True).start()

    def handle_reload_raw(self, episodes):
        def task():
            try:
                for name in episodes:
                    episode = Episode(name)
                    episode.load_pcd(self.raw_dir / "pointcloud")
                    episode.load_pcd_meta(self.raw_dir / "meta")
                    episode.load_skeleton(self.raw_dir / "kinect")
                    self.update_episode(name, episode)
                    self.status_df.loc[name] = [False] * len(self.status_df.columns)
                self.gui_refresh_callback()
            except Exception as e:
                logger.error(f"Failed to reload raw data: {e}")

        threading.Thread(target=task, daemon=True).start()

    def handle_data_align(self, episodes, skeleton_ts_offset_ms: int):
        def task():
            try:
                for name in episodes:
                    episode = self.get_episode(name)
                    aligned = episode.align_traces(
                        use_interp_skel=True,
                        skeleton_ts_type='unix_ms',
                        skeleton_ts_offset_ms=skeleton_ts_offset_ms,
                    )
                    self.update_episode(name, aligned)
                    self.status_df.loc[name, "Aligned"] = True
                self.gui_refresh_callback()
            except Exception as e:
                logger.error(f"Failed to align data: {e}")
                traceback.print_exc()

        threading.Thread(target=task, daemon=True).start()
