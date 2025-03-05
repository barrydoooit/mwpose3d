
from collections import deque
from enum import Enum
import json
import os
import threading
import time
from typing import Any, Callable, Deque, List, Optional
from apps.common.pcd.pointCloud import PointCloudFrame, SimplePointCloud5D
from contextlib import contextmanager


class PointCloudBuffer:
    def __init__(self,
                 max_buffer_size,
                 output_dir: str):
        self.max_buffer_size = max_buffer_size
        self.output_dir = output_dir
        
        self.buffer: Deque[PointCloudFrame] = deque(maxlen=max_buffer_size)
        self._buffer_lock: threading.Lock = threading.Lock()
        @contextmanager
        def locked_buffer():
            with self._buffer_lock:
                yield self.buffer
        self.locked_buffer = locked_buffer
        
        self._frame_entrance_threshold = 0
        self._frame_counter = 0
        self._dump_stage = False
        self.recent_dump = None
        
        self._on_frame_arrival = None
        self._on_buffer_full = None

        self.meta_data = {}
    
    def __getitem__(self, index: int):
        with self._buffer_lock:
            return self.buffer[index]
    
    def __len__(self):
        with self._buffer_lock:
            return len(self.buffer)
    
    def _count(self):
        temp = self._frame_counter
        self._frame_counter += 1
        return temp
    
    def _check_full(self, on_buffer_full: Optional[Callable[["PointCloudBuffer"], Any]] = None) -> bool:
        if self.max_buffer_size == 1:
            # NOTE: If buffer size is 1, we don't need to check if it's full
            return False
        if len(self.buffer) >= self.buffer.maxlen:
            if on_buffer_full:
                on_buffer_full(self)
            return True
        return False
    
    def enlarge_buffer(self, new_size: int) -> int:
        if self.buffer.maxlen is None:
            return
        if new_size <= self.buffer.maxlen:
            return
        with self.locked_buffer() as buffer:
            current_size = len(buffer)
            self.buffer = deque(buffer, maxlen=new_size)
        return current_size
        
    def add_frame(self, point_cloud: SimplePointCloud5D):
        if len(point_cloud.points) <= self._frame_entrance_threshold:
            print(f"Frame entrance threshold not met: {len(point_cloud.points)}")
            return
        
        with self._buffer_lock:
            ts_ms = int(time.time() * 1000)
            frame = PointCloudFrame(self._count(), ts_ms, point_cloud)
            self.buffer.append(frame)
        if self._on_frame_arrival:
            self._on_frame_arrival(self)
            
        self._check_full(self._on_buffer_full)

    def change_frame_entrance_threshold(self, new_threshold: int):
        self._frame_entrance_threshold = new_threshold
    
    def change_max_buffer_size(self, new_size: int):
        with self._buffer_lock:
            self.max_buffer_size = new_size
            self.buffer = deque(self.buffer, maxlen=new_size)
    
    def set_on_frame_arrival(self, callback: Callable[['PointCloudBuffer'], Any]):
        self._on_frame_arrival = callback
    
    def set_on_buffer_full(self, callback: Callable[['PointCloudBuffer'], Any]):
        self._on_buffer_full = callback
    
    def add_metadata(self, key: str, value: Any):
        self.meta_data[key] = value
        return self
    
    def dump(self):
        assert self._check_full()
        with self._buffer_lock:
            self._dump_stage = True
            frames_to_dump = list(self.buffer)
        self.clear()
        
        first_ts = frames_to_dump[0].ts
        last_ts = frames_to_dump[-1].ts
        first_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(first_ts / 1000))
        last_time_str = time.strftime("%H%M%S", time.localtime(last_ts / 1000))
        filename = f"{first_time_str}-{last_time_str}_{len(frames_to_dump)}.json"
        filename = os.path.join(self.output_dir, filename)
        
        data = {
            "frame_keys": ["seq", "ts", "points"],
            "point_keys": ["x", "y", "z", "vel", "snr"],
            "frames": [frame.serialize(compact=True) for frame in frames_to_dump]
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
            
        if len(self.meta_data) > 0:
            meta_dir = os.path.join(self.output_dir, "meta")
            os.makedirs(meta_dir, exist_ok=True)
            meta_filename = os.path.join(meta_dir, f"{first_time_str}-{last_time_str}_{len(frames_to_dump)}.json")
            with open(meta_filename, 'w') as f:
                json.dump(self.meta_data, f)
            
        self.recent_dump = f"{first_time_str}-{last_time_str}_{len(frames_to_dump)}.json"
        print(f"Dumped {len(frames_to_dump)} frames to {filename}")
        
        self._dump_stage = False
    
    def clear(self):
        with self._buffer_lock:
            self.buffer.clear()
            self.buffer = deque(maxlen=self.max_buffer_size)
        self._frame_counter = 0
    
    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(
            max_buffer_size=cfg.get('max_buffer_size', 2000),
            output_dir=cfg.get('output_dir', './data')
        )
        