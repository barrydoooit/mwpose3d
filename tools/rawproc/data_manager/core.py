import logging
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import pickle
import threading
from abc import ABC, abstractmethod
import traceback
from typing import Dict, Optional, Protocol

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from .widgets import CheckList, MultiColumnCheckList
from ...rawproc.episode import Episode
from ...rawproc.hdf5_dumper import ToHdf5



class DataProcessorProtocol(Protocol):
    @property
    @abstractmethod
    def processed_episode_names(self) -> set: ...
    @abstractmethod
    def load_processed_episodes(self, lazy: bool) -> list: ...

    @property
    def episodes(self) -> Dict[str, Episode]: ...
    @abstractmethod
    def get_episode(self, name: str) -> Episode: ...
    @abstractmethod
    def update_episode(self, name: str, new_episode: Episode): ...

    @abstractmethod
    def handle_create_data(self, episodes: list, suffix: str): ...
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
    def handle_data_align(self, episodes: list): ...
    
class DataProcessorGUI(tk.Tk):
    def __init__(self, processor: DataProcessorProtocol):
        super().__init__()
        self.title("Data Processing GUI")
        self.geometry("800x600")
        self.processor = processor
        
        self._buttons = {}
        
        self.create_widgets()
        self.processor.load_processed_episodes()
        self.refresh_episode_lists()
        self.processor._gui_refresh_callabck = self.refresh_episode_lists

    def create_widgets(self):
        # 主布局分为左右两栏
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧episode列表
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        op_columns = ["Calibrated", "Aligned"]
        # 未处理列表
        self.unprocessed_list = MultiColumnCheckList(
            list_frame, "Unprocessed Episodes", self.toggle_buttons_state, op_columns)
        self.unprocessed_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 已处理列表
        self.processed_list = MultiColumnCheckList(
            list_frame, "Processed Episodes", self.toggle_buttons_state, op_columns)
        self.processed_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 右侧控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # 信息后缀输入
        ttk.Label(control_frame, text="Info Suffix:").pack(anchor=tk.W)
        self.info_suffix_entry = ttk.Entry(control_frame)
        self.info_suffix_entry.pack(fill=tk.X, pady=5)

        self.create_control_buttons(control_frame)
            
        # 状态栏
        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var).pack(
            side=tk.BOTTOM, fill=tk.X)
        
    def create_control_buttons(self, master=None):
        button_configs = [
            ("Create Data", self.create_data),
            ("Delete Raw", self.delete_raw),
            ("Allocate to Info", self.allocate_to_info),
            ("Calibrate Time", self.calibrate_time),
            ("Remove Processed", self.remove_processed),
            ("Reload Raw", self.reload_raw),
            ("Align Data", self.align_data)
        ]
        for text, cmd in button_configs:
            self.add_custom_button(master, text, cmd)
    
    def add_custom_button(self, master: Optional[ttk.Frame], text: str, handler: callable):
        new_btn = ttk.Button(self if master is None else master, text=text, command=handler)
        new_btn.pack(fill=tk.X, pady=5)
        self._buttons[text] = new_btn

    def get_selected_episodes(self, lazy=True):
        if lazy:
            return tuple(
                name for name in
                self.unprocessed_list.checked_items + self.processed_list.checked_items
            )
        return {name: self.processor.get_episode(name) for name in self.unprocessed_list.checked_items + self.processed_list.checked_items}
    
    def toggle_buttons_state(self, _=None):
        has_selection = bool(
            self.unprocessed_list.checked_items or 
            self.processed_list.checked_items
        )
        state = tk.NORMAL if has_selection else tk.DISABLED
        for btn in self._buttons.values():
            btn.config(state=state)
    
    # ---- Event handles delegated to processor ----
    def create_data(self):
        episodes = self.get_selected_episodes(lazy=False)
        suffix = self.info_suffix_entry.get().strip()
        self.processor.handle_create_data(episodes, suffix)
    
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
        self.processor.handle_data_align(episode_names)
    # ---- ----

    def update_ui_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in self._buttons.values():
            btn.config(state=state)
    
    def refresh_episode_lists(self):
        radar_dir = self.processor.raw_dir / "radar"
        if radar_dir.exists():
            radar_files = list(radar_dir.glob("*"))
            episode_names = {f.stem for f in radar_files if f.name != "meta"}
        else:
            episode_names = set()
        
        for ep in episode_names:
            if ep not in self.processor.status_df.index:
                self.processor.status_df.loc[ep] = [False] * len(self.processor.status_df.columns)
        processed = sorted(self.processor.processed_episode_names)
        unprocessed = sorted(episode_names - self.processor.processed_episode_names)
        
        self.unprocessed_list.set_items(self.processor.status_df.loc[unprocessed])
        self.processed_list.set_items(self.processor.status_df.loc[processed])
    
    def show_status(self, message: str, error: bool = False):
        fg = "red" if error else "black"
        self.status_var.set(message)
        self.after(10000, lambda: self.status_var.set(""))
    

class DataProcessorDelegate(DataProcessorProtocol):
    def __init__(self, raw_dir: str, output_dir):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self._processed_episode_names = set()
        self._episodes = {}
        self._gui_refresh_callabck = None
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
    
    def load_processed_episodes(self, lazy=True):
        info_path = self.output_dir / "info_all.pkl"
        if info_path.exists():
            with open(info_path, "rb") as f:
                self._processed_episode_names = {
                    Path(name).stem for name in pickle.load(f)
                }
        if lazy:
            return

    def get_episode(self, name: str):
        if name in self.episodes:
            return self.episodes[name]
        episode = Episode(name)
        episode.load_pcd(self.raw_dir / 'radar')
        episode.load_pcd_meta(self.raw_dir / 'radar' / 'meta')
        episode.load_skeleton(self.raw_dir / 'kinect')
        self.episodes[name] = episode
        return episode

    def update_episode(self, name, new_episode):
        assert isinstance(new_episode, Episode)
        self.episodes[name] = new_episode
        logger.info(f"Episode {name} updated.")
    
    def gui_refresh_callback(self):
        if self._gui_refresh_callabck:
            self._gui_refresh_callabck()
    
    def handle_create_data(self, episodes: list, suffix: str):
        def task():
            
                suffixes = ["all"]
                if suffix:
                    suffixes.append(suffix)
                for name in episodes:
                    try:
                        episode = self.get_episode(name)
                        hdf5_maker = ToHdf5(episode, self.output_dir)
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
                    paths = [
                        self.raw_dir / 'radar' / f"{name}.json",
                        self.raw_dir / 'radar' / 'meta' / f"{name}.json",
                        self.raw_dir / 'kinect' / f"{name}.csv"
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
                info_all ={}
                info_pkl_path = self.output_dir / "info_all.pkl"
                if info_pkl_path.exists():
                    with open(info_pkl_path, "rb") as f:
                        info_all = pickle.load(f)
                new_info = {}
                new_info_path = self.output_dir / f"info_{suffix}.pkl"
                if new_info_path.exists():
                    with open(new_info_path, "rb") as f:
                        new_info = pickle.load(f)
                for name in episodes:
                    try:
                        key = f'{name}.h5'
                        if key in info_all:
                            new_info[key] = info_all[key]
                    except Exception as e:
                        logger.error(f"Failed to allocate to info: {e}")            
                with open(new_info_path, "wb") as f:
                    pickle.dump(new_info, f)
            except Exception as e:
                logger.error(f"Failed to allocate to info: {e}")
        threading.Thread(target=task, daemon=True).start()
    
    def handle_calibrate_time(self, episodes: list):
        def task():
            for name in episodes:
                try:
                    episode = self.get_episode(name)
                    calibrated = episode.calibrate_time(manual=True)
                    if calibrated is not None:
                        self.episodes[name] = calibrated
                    self.status_df.loc[name, "Calibrated"] = True
                except Exception as e:
                    logger.error(f"Failed to calibrate time: {e}")
                    traceback.print_exc()
            self.gui_refresh_callback()
        threading.Thread(target=task, daemon=True).start()
    
    def handle_remove_processed(self, episodes, also_data):
        def task():
            try:
                all_info_files = list(self.output_dir.glob("info_*.pkl"))
                for name in episodes:
                    if also_data:
                        paths = [self.output_dir / 'h5' / f"{name}.h5"]
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
                    self._processed_episode_names.remove(name)
                self.gui_refresh_callback()
            except Exception as e:
                logger.error(f"Failed to remove processed data: {e}")
        
        threading.Thread(target=task, daemon=True).start()
    
    def handle_reload_raw(self, episodes):
        def task():
            try:
                for name in episodes:
                    episode = Episode(name)
                    episode.load_pcd(self.raw_dir / 'radar')
                    episode.load_pcd_meta(self.raw_dir / 'radar' / 'meta')
                    episode.load_skeleton(self.raw_dir / 'kinect')
                    self.update_episode(name, episode)
                    self.status_df.loc[name] = [False] * len(self.status_df.columns)
                self.gui_refresh_callback()
            except Exception as e:
                logger.error(f"Failed to reload raw data: {e}")
        threading.Thread(target=task, daemon=True).start()
    
    def handle_data_align(self, episodes):
        def task():
            try:
                for name in episodes:
                    episode = self.get_episode(name)
                    aligned = episode.align_traces(use_interp_skel=True)
                    self.update_episode(name, aligned)
                    self.status_df.loc[name, "Aligned"] = True
                self.gui_refresh_callback()
            except Exception as e:
                logger.error(f"Failed to align data: {e}")
        threading.Thread(target=task, daemon=True).start()