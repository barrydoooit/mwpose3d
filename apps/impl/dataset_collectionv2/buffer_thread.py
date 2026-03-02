from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
from typing import Deque, Optional, Union
from collections import deque
import time
import numpy as np
import struct

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
        self.refusing_new_frames = False

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
    
    def dump_to_json(self, json_file: Union[str, Path], write_meta: bool = True):
        with self.lock() as container:
            frames = list(container)
        meta_data = self._meta_data.copy()

        if not frames:
            return None

        jf = Path(json_file)
        is_dir_like = (not jf.suffix) or (jf.exists() and jf.is_dir())

        if is_dir_like:
            given_dir = jf
            first_ts = frames[0].timestamp
            last_ts = frames[-1].timestamp
            first_time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(first_ts / 1000))
            last_time_str  = time.strftime('%H%M%S',        time.localtime(last_ts  / 1000))
            out_path = given_dir / f"{first_time_str}-{last_time_str}_f{len(frames)}.json"
        else:
            out_path = jf
            given_dir = out_path.parent  # the directory that will hold the trace file
        given_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "frame_keys": ['seq', 'ts', 'points'],
            "point_keys": ['x', 'y', 'z', 'vel', 'snr'],
            "frames": [frame.serialize(compact=True) for frame in frames],
        }
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp_path, out_path)
        if write_meta and meta_data:
            meta_dir = given_dir.parent / 'meta'
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = meta_dir / out_path.with_suffix('.meta.json').name
            tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
            with open(tmp_meta, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f)
            os.replace(tmp_meta, meta_path)
        self.clear()
        print(str(out_path))
        return str(out_path)


class RawBinaryBuffer:
    def __init__(self, max_buffer_size: int = 1000, save_with_timestamp: bool = False):
        self.max_buffer_size = max_buffer_size
        self._container: Deque[tuple[float, bytes]] = deque(maxlen=max_buffer_size)
        self._frame_counter = 0
        self._meta_data = dict()
        self.save_with_timestamp = bool(save_with_timestamp)

    @property
    def buffer_lock(self) -> threading.Lock:
        if not hasattr(self, '_buffer_lock'):
            self._buffer_lock = threading.Lock()
        return self._buffer_lock

    @contextmanager
    def lock(self):
        with self.buffer_lock:
            yield self._container

    def __len__(self) -> int:
        with self.lock() as container:
            return len(container)

    def _counter_increment(self):
        self._frame_counter += 1

    def _check_full(self) -> bool:
        return len(self) >= self.max_buffer_size

    def append(self, raw_bytes: Union[bytes, bytearray, memoryview], timestamp: Optional[float] = None):
        if self._check_full():
            return
        ts_ms = float(timestamp) if timestamp is not None else float(int(time.time() * 1000))
        payload = bytes(raw_bytes)
        with self.lock() as container:
            container.append((ts_ms, payload))
            self._counter_increment()

    def suggest_filename(self, suffix: str = ".bin") -> Optional[str]:
        with self.lock() as container:
            frames = list(container)
        if not frames:
            return None
        first_ts = frames[0][0]
        last_ts = frames[-1][0]
        first_time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(first_ts / 1000))
        last_time_str = time.strftime('%H%M%S', time.localtime(last_ts / 1000))
        return f"{first_time_str}-{last_time_str}_f{len(frames)}{suffix}"

    def record_meta(self, key: str, value: any):
        self._meta_data[key] = value

    def clear(self):
        with self.lock() as container:
            container.clear()
        self._frame_counter = 0
        self._meta_data = dict()

    def dump_to_bin(self, bin_file: Union[str, Path]):
        with self.lock() as container:
            frames = list(container)
        meta_data = self._meta_data.copy()

        if not frames:
            return None

        bf = Path(bin_file)
        is_dir_like = (not bf.suffix) or (bf.exists() and bf.is_dir())
        if is_dir_like:
            given_dir = bf
            first_ts = frames[0][0]
            last_ts = frames[-1][0]
            first_time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(first_ts / 1000))
            last_time_str = time.strftime('%H%M%S', time.localtime(last_ts / 1000))
            out_path = given_dir / f"{first_time_str}-{last_time_str}_f{len(frames)}.bin"
        else:
            out_path = bf
            given_dir = out_path.parent

        given_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, 'wb') as f:
            for ts_ms, raw_bytes in frames:
                if self.save_with_timestamp:
                    f.write(struct.pack("<d", float(ts_ms)))
                f.write(raw_bytes)
        os.replace(tmp_path, out_path)

        if meta_data:
            meta_dir = given_dir.parent / 'meta'
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = meta_dir / out_path.with_suffix('.meta.json').name
            tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
            with open(tmp_meta, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f)
            os.replace(tmp_meta, meta_path)

        self.clear()
        print(str(out_path))
        return str(out_path)
    
class PointCloudBufferingWorker(QObject):
    bufferFull = Signal()
    bufferDumped = Signal(str) # dumped file name (e.g., .json or .bin)
    recordMeta = Signal(dict)
    frameCount = Signal(int)

    def __init__(
        self,
        dump_dir: Union[str, Path],
        buffer_size: int = 1000,
        storage_format: str = 'pointcloud_json',
        pointcloud_dump_dir: Optional[Union[str, Path]] = None,
        save_with_timestamp: bool = False,
    ):
        super().__init__()
        self.storage_format = storage_format
        if self.storage_format == 'pointcloud_json':
            self.buffer = PointCloudBuffer(max_buffer_size=buffer_size)
        elif self.storage_format == 'raw_bin':
            self.buffer = RawBinaryBuffer(
                max_buffer_size=buffer_size,
                save_with_timestamp=save_with_timestamp,
            )
        elif self.storage_format == 'raw_bin_and_pointcloud_json':
            self.raw_buffer = RawBinaryBuffer(
                max_buffer_size=buffer_size,
                save_with_timestamp=save_with_timestamp,
            )
            self.pcd_buffer = PointCloudBuffer(max_buffer_size=buffer_size)
            self.buffer = self.raw_buffer  # backward-compatible fallback accessor
        else:
            raise ValueError(
                f"Unsupported storage_format '{self.storage_format}'. "
                "Expected one of ['pointcloud_json', 'raw_bin', 'raw_bin_and_pointcloud_json']."
            )
        self.dump_dir = Path(dump_dir)
        if self.storage_format == 'raw_bin_and_pointcloud_json':
            if pointcloud_dump_dir is None:
                self.pointcloud_dump_dir = self.dump_dir.parent / 'pointcloud'
            else:
                self.pointcloud_dump_dir = Path(pointcloud_dump_dir)
        else:
            self.pointcloud_dump_dir = None
        self.refusing_new_frames = False

    def _buffer_length(self) -> int:
        if self.storage_format == 'raw_bin_and_pointcloud_json':
            return len(self.raw_buffer)
        return len(self.buffer)

    def _buffer_full(self) -> bool:
        if self.storage_format == 'raw_bin_and_pointcloud_json':
            return self.raw_buffer._check_full()
        return self.buffer._check_full()

    def _emit_count_and_full_if_needed(self):
        length = self._buffer_length()
        self.frameCount.emit(length)
        if self._buffer_full():
            self.bufferFull.emit()

    @staticmethod
    def _radar_frame_to_simple_pcd(frame: object) -> SimplePointCloud5D:
        point_cloud = getattr(frame, 'point_cloud', None)
        if point_cloud is None:
            return SimplePointCloud5D([])

        arr = np.asarray(point_cloud)
        if arr.ndim != 2:
            return SimplePointCloud5D([])
        # Accept both (6, N)/(5, N) and (N, 6)/(N, 5)
        if arr.shape[0] in (5, 6):
            arr = arr.T
        elif arr.shape[1] not in (5, 6):
            return SimplePointCloud5D([])

        if arr.shape[1] < 5:
            return SimplePointCloud5D([])
        return SimplePointCloud5D.from_numpy(arr[:, :5])

    @Slot(object, float)
    def enqueue(self, point_cloud: 'SimplePointCloud5D', timestamp: Optional[float] = None):
        if self.storage_format not in ('pointcloud_json', 'raw_bin_and_pointcloud_json'):
            return
        if self.storage_format == 'raw_bin_and_pointcloud_json':
            self.pcd_buffer.append(point_cloud, timestamp)
        else:
            self.buffer.append(point_cloud, timestamp)
        self._emit_count_and_full_if_needed()

    @Slot(object)
    def enqueue_frame(self, frame: object):
        if self.refusing_new_frames or self.storage_format not in ('raw_bin', 'raw_bin_and_pointcloud_json'):
            return

        raw_bytes = None
        timestamp = None
        if hasattr(frame, 'raw_bytes'):
            raw_bytes = getattr(frame, 'raw_bytes', None)
            timestamp = getattr(frame, 'frame_start_timestamp_ms', None)
        elif isinstance(frame, dict):
            raw_bytes = frame.get('raw_bytes', None)
            timestamp = frame.get('timestamp', None)
        elif isinstance(frame, (bytes, bytearray, memoryview)):
            raw_bytes = frame

        if raw_bytes is None:
            return

        if self.storage_format == 'raw_bin_and_pointcloud_json':
            pcd = self._radar_frame_to_simple_pcd(frame)
            self.raw_buffer.append(raw_bytes, timestamp)
            self.pcd_buffer.append(pcd, timestamp)
        else:
            self.buffer.append(raw_bytes, timestamp)
        self._emit_count_and_full_if_needed()
    
    @Slot(dict)
    def enqueue_raw(self, data: dict):
        if self.refusing_new_frames:
            return
        if self.storage_format in ('raw_bin', 'raw_bin_and_pointcloud_json'):
            raw_bytes = data.get('raw_bytes', None)
            if raw_bytes is None:
                return
            timestamp = data.get('timestamp', None)
            if self.storage_format == 'raw_bin_and_pointcloud_json':
                self.raw_buffer.append(raw_bytes, timestamp)
                try:
                    pcd = SimplePointCloud5D.from_dict(data.copy())
                except Exception:
                    pcd = SimplePointCloud5D([])
                self.pcd_buffer.append(pcd, timestamp)
            else:
                self.buffer.append(raw_bytes, timestamp)
            self._emit_count_and_full_if_needed()
            return
        point_cloud = SimplePointCloud5D.from_dict(data)
        timestamp = data.get('timestamp', None)
        self.enqueue(point_cloud, timestamp)

    @Slot(str)
    def dump_buffer(self, json_file: Optional[str] = None):
        self.refusing_new_frames = True
        dump_path = self.dump_dir / json_file if json_file else self.dump_dir
        if self.storage_format == 'pointcloud_json':
            final_path = self.buffer.dump_to_json(dump_path)
        elif self.storage_format == 'raw_bin':
            final_path = self.buffer.dump_to_bin(dump_path)
        else:
            if json_file:
                stem = Path(json_file).stem
                raw_name = f"{stem}.bin"
            else:
                raw_name = self.raw_buffer.suggest_filename(".bin")
            if raw_name is None:
                final_path = None
            else:
                raw_path = self.dump_dir / raw_name
                pcd_path = self.pointcloud_dump_dir / f"{Path(raw_name).stem}.json"
                final_path = self.raw_buffer.dump_to_bin(raw_path)
                # Meta is already written by raw dump using the shared stem.
                self.pcd_buffer.dump_to_json(pcd_path, write_meta=False)
        if final_path is not None:
            self.bufferDumped.emit(Path(final_path).name)
        self.refusing_new_frames = False
    
    @Slot(dict)
    def record_meta(self, meta_data: dict):
        if self.storage_format == 'raw_bin_and_pointcloud_json':
            for key, value in meta_data.items():
                self.raw_buffer.record_meta(key, value)
                self.pcd_buffer.record_meta(key, value)
            return
        for key, value in meta_data.items():
            self.buffer.record_meta(key, value)

    @Slot()
    def clear_buffer(self):
        if self.storage_format == 'raw_bin_and_pointcloud_json':
            self.raw_buffer.clear()
            self.pcd_buffer.clear()
            return
        self.buffer.clear()
    
    def run(self):
        self.exec_()

    @classmethod
    def build_with_thread(cls, **kwargs):
        dump_dir = kwargs.get('dump_dir', None)
        if dump_dir is None:
            raise ValueError("Missing directory to store point cloud traces.")
        buffer_size = kwargs.get('buffer_size', 1000)
        storage_format = kwargs.get('storage_format', 'pointcloud_json')
        pointcloud_dump_dir = kwargs.get('pointcloud_dump_dir', None)
        save_with_timestamp = kwargs.get('save_with_timestamp', False)

        thread = QThread()
        worker = cls(
            dump_dir=dump_dir,
            buffer_size=buffer_size,
            storage_format=storage_format,
            pointcloud_dump_dir=pointcloud_dump_dir,
            save_with_timestamp=save_with_timestamp,
        )
        worker.moveToThread(thread)
        worker.recordMeta.connect(worker.record_meta)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        return worker, thread
