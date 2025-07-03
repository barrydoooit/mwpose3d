from contextlib import contextmanager
import json
from pathlib import Path
import threading
from typing import Deque, Optional, Union
from collections import deque
import time

from PySide6.QtCore import (
    QThread,
    QObject,
    Signal,
    Slot,
)

from mwpose3d.utils.pointcloud_toolkits.structures import PointCloudFrame, SimplePointCloud5D



class PointCloudBuffer:
    def __init__(self,
                 max_buffer_size: int = 1000):
        self.max_buffer_size = max_buffer_size
        self._container: Deque[PointCloudFrame] = deque(maxlen=max_buffer_size)

        self._frame_counter = 0   

        self._meta_data = dict()

    @property
    def buffer_lock(self) -> threading.Lock:
        if not hasattr(self, '_buffer_lock'):
            self._buffer_lock = threading.Lock()
        return self._buffer_lock
    
    @contextmanager
    def lock(self):
        with self.buffer_lock:
            yield self._container
    
    def __getitem__(self, index: int) -> PointCloudFrame:
        with self.lock() as container:
            return container[index]
    
    def __len__(self) -> int:
        with self.lock() as container:
            return len(container)
    
    def _counter_increment(self):
        self._frame_counter += 1
    
    def _check_full(self) -> bool:
        if len(self) >= self.max_buffer_size:
            return True
        return False
    
    def expand(self, new_size: int):
        if self._container.maxlen is None or new_size > self._container.maxlen:
            return
        with self.lock() as container:
            self._container = deque(container, maxlen=new_size)
    
    def append(self, point_cloud: SimplePointCloud5D, timestamp: float = None):
        if self._check_full():
            return 
        if isinstance(point_cloud, PointCloudFrame):
            with self.lock() as container:
                if timestamp is not None:
                    point_cloud.timestamp = timestamp
                container.append(point_cloud)
                self._counter_increment()
            return
        
        with self.lock() as container:
            ts_ms = timestamp if timestamp is not None else int(time.time() * 1000)
            frame = PointCloudFrame.from_pcd(point_cloud, self._frame_counter, ts_ms)
            container.append(frame)
            self._counter_increment()
        return
    
    def record_meta(self, key: str, value: any):
        self._meta_data[key] = value

    def clear(self):
        with self.lock() as container:
            container.clear()
        self._frame_counter = 0
        self._meta_data = dict()
    
    def dump_to_json(self, json_file: Union[str, Path]):
        with self.lock() as container:
            frames = list(container)
        meta_data = self._meta_data.copy()
        self.clear()

        json_path = Path(json_file)
        if json_path.is_dir():
            first_ts = frames[0].timestamp
            last_ts = frames[-1].timestamp
            first_time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(first_ts / 1000))
            last_time_str = time.strftime('%H%M%S', time.localtime(last_ts / 1000))
            json_path = json_path / f"{first_time_str}-{last_time_str}_f{len(frames)}.json"

        if not json_path.parent.exists():
            json_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "frame_keys": ['seq', 'ts', 'points'],
            "point_keys": ['x', 'y', 'z', 'vel', 'snr'],
            "frames": [frame.serialize(compact=True) for frame in frames],
        }
        with open(json_path, 'w') as f:
            json.dump(data, f)

        if len(meta_data) > 0:
            meta_path = json_path.with_suffix('.meta.json')
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f)
        
        return str(json_path)
    
class PointCloudBufferingWorker(QObject):
    bufferFull = Signal()
    bufferDumped = Signal(str) # json file path
    recordMeta = Signal(dict)
    frameCount = Signal(int)

    def __init__(self, dump_dir: Union[str, Path], buffer_size: int = 1000):
        super().__init__()
        self.buffer = PointCloudBuffer(max_buffer_size=buffer_size)
        self.dump_dir = Path(dump_dir)
    
    @Slot(object, float)
    def enqueue(self, point_cloud: 'SimplePointCloud5D', timestamp: Optional[float] = None):
        self.buffer.append(point_cloud, timestamp)
        length = len(self.buffer)
        self.frameCount.emit(length)
        if self.buffer._check_full():
            self.bufferFull.emit()
    
    @Slot(dict)
    def enqueue_raw(self, data: dict):
        point_cloud = SimplePointCloud5D.from_dict(data)
        timestamp = data.get('timestamp', None)
        self.enqueue(point_cloud, timestamp)

    @Slot(str)
    def dump_buffer(self, json_file: Optional[str] = None):
        dump_path = self.dump_dir / json_file if json_file else self.dump_dir
        final_path = self.buffer.dump_to_json(dump_path)
        self.bufferDumped.emit(final_path)
    
    @Slot(dict)
    def record_meta(self, meta_data: dict):
        for key, value in meta_data.items():
            self.buffer.record_meta(key, value)

    @Slot()
    def clear_buffer(self):
        self.buffer.clear()
    
    def run(self):
        self.exec_()

    @classmethod
    def build_with_thread(cls, **kwargs):
        dump_dir = kwargs.get('dump_dir', None)
        if dump_dir is None:
            raise ValueError("Missing directory to store point cloud traces.")
        buffer_size = kwargs.get('buffer_size', 1000)

        thread = QThread()
        worker = cls(dump_dir=dump_dir, buffer_size=buffer_size)
        worker.moveToThread(thread)
        worker.recordMeta.connect(worker.record_meta)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        return worker, thread