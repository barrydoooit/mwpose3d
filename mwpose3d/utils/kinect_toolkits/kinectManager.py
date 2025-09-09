import glob
import os
from copy import copy
import subprocess
from pathlib import Path
import time
from typing import List, Optional, Literal
import logging
logger = logging.getLogger(__name__)


from .kinectData import Skeleton
from .kinectParser import get_last_record

class KinectManager:
    FILE_PREFIX = "CURRENT"
    def __init__(self,
                 exe_path: str,
                 output_dir: str,
                 mode: List[Literal["dump", "capture", "control"]] = ["capture", "control"]
                 ):
        self.exe_path = os.path.abspath(exe_path)
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            logger.info(f"Output directory {self.output_dir} does not exist. Creating it.")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_prefix = Path(output_dir) / self.FILE_PREFIX
        self.process = None
        self.mode = copy(mode)

        self.capture_mmf_reader = CaptureMmfReader() if self.is_capturing else None

        logger.info(f"Kinect manager initialized with exe_path: {self.exe_path}, output_dir: {self.output_dir}")
    
    def refresh(self):
        if self.process is not None:
            self.stop_skeleton_capture(wait=2)
        if self.capture_mmf_reader is not None:
            self.capture_mmf_reader.close()
            self.capture_mmf_reader = CaptureMmfReader() if self.is_capturing else None
    
    @property
    def is_dumping_to_file(self) -> bool:
        return "dump" in self.mode
    
    @property
    def is_capturing(self) -> bool:
        return "capture" in self.mode
    
    @property
    def is_controller(self) -> bool:
        return "control" in self.mode
    
    def error_if_not_dumping(self):
        logger.error("KinectManager is indicated to not dump to file. " \
            "Add 'dump' to the mode list when you have your exe ready for dumping to file.")
        return None
    
    def error_if_not_capturing(self):
        logger.error("KinectManager is indicated to not capture. " \
            "Add 'capture' to the mode list when you have your exe ready for capturing.")
        return None
    
    def start_skeleton_capture(self):
        if self.process is not None:
            raise RuntimeError("Kinect process is already running")
        logger.info("Starting Kinect process...")
        if self.is_dumping_to_file:
            self.delete_default_output_file()
        cmd = [self.exe_path, "--prefix", str(self.output_prefix)]
        if not self.is_capturing:
            cmd.append("--ncap")
        if not self.is_controller:
            cmd.append("--nctrl")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Kinect process started with pid: {self.process.pid}")
    
    def stop_skeleton_capture(self, wait=2):
        if self.process is None:
            raise RuntimeError("Kinect process is not running")
        time.sleep(wait)
        logger.info("Waiting for Kinect process to terminate...")
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Forcefully killing the process...")
            self.process.kill()
        logger.info("Kinect process stopped.")
        if self.capture_mmf_reader is not None:
            self.capture_mmf_reader.close()
        self.process = None
    
    def get_default_output_file(self, datatype="body"):
        if not self.is_dumping_to_file: return self.error_if_not_dumping()
        if datatype == "body":
            pattern = str(self.output_dir / f"{self.FILE_PREFIX}*.csv")
        else:
            raise NotImplementedError(f"Unknown datatype: {datatype}")
        matches = glob.glob(pattern)
        if len(matches) == 0:
            return None
        return Path(matches[0])
    
    def rename_default_output_file(self, new_name: str):
        if not self.is_dumping_to_file: return self.error_if_not_dumping()
        self._final_file_name = new_name
        if self._final_file_name is None:
            logger.info(f"Kinect file stored at default path: {self.get_default_output_file()}")
            return
        new_file = self.output_dir / self._final_file_name
        if new_file.suffix.lower() != ".csv":
            new_file = new_file.with_suffix(".csv")
        
        try:
            self.get_default_output_file().rename(new_file)
            self._final_file_name = None
            logger.info("Kinect file stored at path: ", str(new_file))
        except Exception as e:
            logger.error("Error while renaming the file: ", e)
    
    def delete_default_output_file(self):
        if not self.is_dumping_to_file: return self.error_if_not_dumping()
        file_path = self.get_default_output_file()
        if file_path is not None and file_path.exists():
            logger.info("Deleting file: ", file_path)
            file_path.unlink()
            
    def wait_for_capture_starts(self):
        if self.is_dumping_to_file:
            while True:
                file_path = self.get_default_output_file()
                if file_path is not None and file_path.exists():
                    if file_path.stat().st_size > 1024:
                        return True
                time.sleep(0.2)
        
        if self.is_capturing:
            self.capture_mmf_reader.wait_for_frame(timeout=None)
            logger.info("First frame received, capture started.")
            return True
            
    def get_last_record(self) -> Optional[Skeleton]:
        file_path = self.get_default_output_file()
        if file_path is None:
            return None
        return get_last_record(file_path)

    def get_new_records(self, to_radar_coord: bool = True) -> List[Skeleton]:
        if not self.is_capturing:
            return self.error_if_not_capturing()
        
        records: list[Skeleton] = []
        for frame in self.capture_mmf_reader.read_frames():
            joints = frame["joints"]
            skeleton = Skeleton.from_sequence(
                timestamp=frame["rel_time"],
                unix_ms=frame["unix_ms"],
                sequence=[coord for joint in joints for coord in joint["position"]],
            )
            records.append(skeleton)
        if to_radar_coord:
            for record in records:
                record.transpose_(0, 2, 1)
                for joint in record.keypoints:
                    record.keypoints[joint].x *= -1  # Invert x-axis for radar coordinates

        return records

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None
    
    @classmethod
    def from_dict(cls, cfg: dict):
        exe_path = cfg.get("exe_path", os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "DumpKinectSkeleton",
                "bin",
                "Release",
                "DumpKinectSkeleton.exe"
            )
        ))
        output_dir = cfg.get("output_dir")
        return cls(
            exe_path=exe_path,
            output_dir=output_dir,
            mode=cfg.get("mode", ["capture", "control"])
        )

import mmap
import struct

class CaptureMmfReader:
    def __init__(self,
                 tagname: str = r"Local\KinectCapture",
                 num_joints: int = 25,
                 ring_depth: int = 128):
        self._nj = num_joints
        self._d = ring_depth
        self._packet_size = (8 + 8 + self._nj * (3*4 + 4*4 + 4))
        self._total_size = 4 + self._d * self._packet_size

        self._mm = mmap.mmap(
            -1,
            self._total_size,
            tagname=tagname,
            access=mmap.ACCESS_WRITE
        )
        self._last_id = None

    def read_frames(self):
        self._mm.seek(0)
        head_id = struct.unpack_from("<i", self._mm.read(4), 0)[0]
        if self._last_id is None:
            start = head_id
        else:
            start = self._last_id + 1
        
        end = head_id
        for frame_id in range(start, end + 1):
            slot = frame_id % self._d
            pos = 4 + slot * self._packet_size
            self._mm.seek(pos)
            raw = self._mm.read(self._packet_size)

            relMs, unixMs = struct.unpack_from("<dd", raw, 0)
            joints = []
            offset = 8 + 8
            fmt = "<" + "f" * self._nj * (3 + 4 + 1)
            floats = struct.unpack_from(fmt, raw, offset)

            idx = 0
            for _ in range(self._nj):
                x, y, z = floats[idx: idx + 3]; idx += 3
                qx, qy, qz, qw = floats[idx: idx + 4]; idx += 4
                state = floats[idx]; idx += 1
                joints.append({
                    "position": (x, y, z),
                    "orientation": (qx, qy, qz, qw),
                    "tracking_state": state
                })
            yield {
                "frame_id": frame_id,
                "rel_time": relMs,
                "unix_ms": unixMs,
                "joints": joints
            }
        self._last_id = head_id
    
    def wait_for_frame(self, timeout=None):
        start = time.time()
        while True:
            self._mm.seek(0)
            raw = self._mm.read(4)
            if len(raw) < 4:
                if timeout is not None and time.time() - start > timeout:
                    return False
                time.sleep(0.01)
                continue

            head = struct.unpack_from("<i", raw, 0)[0]

            if self._last_id is None:
                if head > 0:
                    return True
            else:
                if head != self._last_id:
                    return True
            
            if timeout is not None and time.time() - start > timeout:
                return False
            
            time.sleep(0.01)

    def close(self):
        if self._mm is not None:
            self._mm.close()
        self._last_id = None
