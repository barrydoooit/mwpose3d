
from collections import deque
from enum import Enum
import json
import os
import threading
import time
from typing import Any, Callable, Deque, List
from apps.common.pcd.pointCloud import PointCloudFrame, SimplePointCloud5D


class PointCloudBuffer:
    def __init__(self,
                 max_buffer_size,
                 output_dir: str):
        self.max_buffer_size = max_buffer_size
        self.output_dir = output_dir
        
        self.buffer: Deque[PointCloudFrame] = deque(maxlen=max_buffer_size)
        self._buffer_lock: threading.Lock = threading.Lock()
        self._frame_counter = 0
        self._dump_stage = False
        
        self._on_frame_arrival = None
        self._on_buffer_full = None

    
    def __getitem__(self, index: int):
        with self._buffer_lock:
            return self.buffer[index]
            
    def _count(self):
        temp = self._frame_counter
        self._frame_counter += 1
        return temp
    
    
    def _check_full(self) -> bool:
        if self.max_buffer_size == 1:
            # NOTE: If buffer size is 1, we don't need to check if it's full
            return False
        if len(self.buffer) >= self.max_buffer_size:
            if self._on_buffer_full:
                self._on_buffer_full(self)
            return True
        return False
    
    def add_frame(self, point_cloud: SimplePointCloud5D):
        with self._buffer_lock:
            ts_ms = int(time.time() * 1000)
            frame = PointCloudFrame(self._count(), ts_ms, point_cloud)
            self.buffer.append(frame)
        if self._on_frame_arrival:
            self._on_frame_arrival(self)
            
        self._check_full()

    def change_max_buffer_size(self, new_size: int):
        with self._buffer_lock:
            self.max_buffer_size = new_size
            self.buffer = deque(self.buffer, maxlen=new_size)
    
    def set_on_frame_arrival(self, callback: Callable[['PointCloudBuffer'], Any]):
        self._on_frame_arrival = callback
    
    def set_on_buffer_full(self, callback: Callable[['PointCloudBuffer'], Any]):
        self._on_buffer_full = callback
        
    def dump(self):
        assert len(self.buffer) == self.max_buffer_size
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
        # with open(filename, 'w') as f:
        #     json.dump(data, f)
        print(f"Dumped {len(frames_to_dump)} frames to {filename}")
        
        self._dump_stage = False
    
    def clear(self):
        with self._buffer_lock:
            self.buffer.clear()
        self._frame_counter = 0
    
    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(
            max_buffer_size=cfg.get('max_buffer_size', 2000),
            output_dir=cfg.get('output_dir', './data')
        )
        