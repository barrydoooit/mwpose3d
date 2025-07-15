import mmap
import struct
import threading
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition, Slot, QObject
import numpy as np
from typing import TYPE_CHECKING

from mwpose3d.utils.kinect_toolkits.kinectData import KeypointType
if TYPE_CHECKING:
    from .online_skeletion_estim import OnlineSkeletionEstimationApp
    from mwpose3d.runner.inference_engine import InferenceEngine




class InferenceWorkerThread(QThread):
    inferece_done = Signal(np.ndarray)

    _default_ctrl_joints = [
         KeypointType.SPINE_MID,
         KeypointType.ELBOW_RIGHT,
         KeypointType.WRIST_RIGHT,
         KeypointType.ELBOW_LEFT,
         KeypointType.WRIST_LEFT,
     ]
    
    def __init__(self,  
                 app: 'OnlineSkeletionEstimationApp',
                 with_ctrl_joints: bool = True,
                 ctrl_joints: list[int] = None,
                 parent=None):
        super().__init__(parent)
        self.app = app
        self.ctrl_joints = ctrl_joints
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._latest_frame = None
        self._running = False

        self.with_ctrl_joints = with_ctrl_joints
        if self.with_ctrl_joints:
            self.update_ctrl_joints(ctrl_joints)
        _controller_map_name = r"Local\KinectControl"
        self._controller_mmf = mmap.mmap(
            -1, self._controller_packet_size, tagname=_controller_map_name, access=mmap.ACCESS_WRITE
        )
        self._controller_lock = threading.Lock()
        
    def update_ctrl_joints(self, ctrl_joints: list[KeypointType]):
        if ctrl_joints is None or len(ctrl_joints) == 0:
            self._ctrl_joints = self._default_ctrl_joints
        else:
            self._ctrl_joints = [KeypointType(kp) for kp in ctrl_joints]
        self._controller_packet_size = 3 * 4 * len(self._ctrl_joints)
    
    @property
    def inference_engine(self) -> 'InferenceEngine':
        return self.app.inference_engine

    @Slot(object)
    def enqueue(self, frame):
        self._mutex.lock()
        self._latest_frame = frame
        self._cond.wakeOne()
        self._mutex.unlock()
    
    def write_controller_mmf(self, result: np.ndarray):
        coords = []
        keypoints_in_result = self.inference_engine.keypoints_involved
        for kp in self._ctrl_joints:
            idx = keypoints_in_result.index(kp.value)
            x, y, z = result[3*idx:3*idx+3]
            coords.extend([-x, z, y])
        buf = struct.pack(f'{len(coords)}f', *coords)
        with self._controller_lock:
            self._controller_mmf.seek(0)
            self._controller_mmf.write(buf)

    def run(self):
        self._running = True
        while True:
            self._mutex.lock()

            while self._running and self._latest_frame is None:
                self._cond.wait(self._mutex)
            
            if not self._running:
                self._mutex.unlock()
                break

            frame = self._latest_frame
            self._latest_frame = None
            self._mutex.unlock()

            result = self.inference_engine.infer(frame)
            if result is not None:
                self.write_controller_mmf(result)
                self.inferece_done.emit(result)
        
        QThread.currentThread().quit()
    
    def stop(self):
        self._mutex.lock()
        self._running = False
        self._cond.wakeOne()
        self._mutex.unlock()
        self.wait()
