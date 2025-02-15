import glob
import os
import subprocess
from pathlib import Path
import time
from typing import List, Optional


from .kinectData import Skeleton
from .kinectParser import get_last_record

class KinectManager:
    FILE_PREFIX = "CURRENT"
    def __init__(self,
                 exe_path: str,
                 output_dir: str,
                 ):
        self.exe_path = os.path.abspath(exe_path)
        self.output_dir = Path(output_dir)
        self.output_prefix = Path(output_dir) / self.FILE_PREFIX
        self.process = None
        print(f"Kinect manager initialized with exe_path: {self.exe_path}, output_dir: {self.output_dir}")
    
    def start_skeleton_capture(self):
        if self.process is not None:
            raise RuntimeError("Kinect process is already running")
        print("Starting Kinect process...")
        cmd = [self.exe_path, "--prefix", str(self.output_prefix)]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Kinect process started with pid: {self.process.pid}")
    
    def stop_skeleton_capture(self, wait=1):
        if self.process is None:
            raise RuntimeError("Kinect process is not running")
        time.sleep(wait)
        self.process.terminate()
        try:
            print("Waiting for process to terminate...")
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Forcefully killing the process...")
            self.process.kill()
        print("Kinect process stopped.")
        self.process = None
    
    def get_default_output_file(self, datatype="body"):
        if datatype == "body":
            pattern = str(self.output_dir / f"{self.FILE_PREFIX}*.csv")
        else:
            raise NotImplementedError(f"Unknown datatype: {datatype}")
        matches = glob.glob(pattern)
        if len(matches) == 0:
            return None
        return Path(matches[0])
    
    def rename_default_output_file(self, new_name: str):
        self._final_file_name = new_name
        if self._final_file_name is None:
            print(f"Kinect file stored at default path: {self.get_default_output_file()}")
            return
        new_file = self.output_dir / self._final_file_name
        if new_file.suffix.lower() != ".csv":
            new_file = new_file.with_suffix(".csv")
        
        try:
            self.get_default_output_file().rename(new_file)
            self._final_file_name = None
            print("Kinect file stored at path: ", str(new_file))
        except Exception as e:
            print("Error while renaming the file: ", e)
    
    def delete_default_output_file(self):
        file_path = self.get_default_output_file()
        if file_path.exists():
            file_path.unlink()
            
    def wait_for_capture_starts(self):
        
        while True:
            file_path = self.get_default_output_file()
            if file_path is not None and file_path.exists():
                if file_path.stat().st_size > 1024:
                    return True
            time.sleep(0.2)
    
    def get_last_record(self) -> Optional[Skeleton]:
        file_path = self.get_default_output_file()
        if file_path is None:
            return None
        return get_last_record(file_path)

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
                "Debug",
                "DumpKinectSkeleton.exe"
            )
        ))
        output_dir = cfg.get("output_dir")
        return cls(
            exe_path=exe_path,
            output_dir=output_dir,
        )