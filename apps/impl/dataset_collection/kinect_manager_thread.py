import csv
from pathlib import Path
import threading
from typing import Deque, List, Optional, Union
from collections import deque
import time
import logging

from typing import Tuple
logger = logging.getLogger(__name__)
from PySide6.QtCore import (
    QThread,
    QObject,
    Signal,
    Slot,
)

import mwpose3d.utils.kinect_toolkits as ktk


class KinectManagerWorker(QObject):
    # --- Outgoing Signals ---
    recentSkeletonJointCoordSignal = Signal(list)
    captureProcessStartedSignal = Signal()
    dumpFinishedSignal = Signal(str)

    # --- Control Signals ---
    startSkeletonCaptureSignal = Signal()
    pauseSkeletonCaptureSignal = Signal()
    dumpSkeletonsSignal = Signal(str)
    resumeSkeletonCaptureSignal = Signal()
    terminateSkeletonCaptureSignal = Signal()
    
    def __init__(self,
                 kinect_mgr_cfg: dict,):
        super().__init__()
        self.kinect_mgr = ktk.KinectManager.from_dict(kinect_mgr_cfg)

        self.skeletons: List[ktk.Skeleton] = []
        self._running = False
        self._paused = False
    
    @Slot()
    def _on_start(self):
        self._ct = threading.Thread(target=self._run_capture_loop, name="KinectCaptureLoop", daemon=True)
        self._ct.start()
    
    def _run_capture_loop(self):
        try:
            self.kinect_mgr.start_skeleton_capture()
            self.kinect_mgr.wait_for_capture_starts()
            self.captureProcessStartedSignal.emit()
        except Exception as e:
            logger.error(f"Error starting Kinect capture: {e}")
            return

        self._running = self.kinect_mgr.running
        self._paused = False
        
        while self._running and not QThread.currentThread().isInterruptionRequested():
            if self._paused:
                time.sleep(0.1)
                continue
            try:
                new = self.kinect_mgr.get_new_records()
            except Exception as e:
                logger.error(f"Error getting new records: {e}")
                break

            if new:
                self.skeletons.extend(new)
                self.recentSkeletonJointCoordSignal.emit(self.skeletons[-1].flatten()[1][2:])  # Emit only the flattened data excluding timestamp and unix_ms
            
            time.sleep(0.01)

    @Slot()
    def _on_pause(self):
        if not self._running:
            logger.warning("Pause ignored: Kinect capture is not running.")
            return
        self._paused = True
        logger.info("Paused reading from Kinect Capture process.")
    
    @Slot()
    def _on_resume(self):
        self._paused = False
        logger.info("Resumed reading from Kinect Capture process.")
    
    @Slot()
    def _on_stop(self):
        if self._ct.is_alive():
            logger.info("Stopping Kinect Capture thread.")
            self._ct.join(timeout=1)
        self.kinect_mgr.refresh()
        self._running = False
        self._paused = False
        logger.info("Kinect Capture thread terminated.")

    @Slot(str)
    def _on_dump(self, filename: str):
        if len(self.skeletons) == 0:
            logger.warning("No skeletons to dump.")
            return
        if not self._paused:
            self.pauseSkeletonCaptureSignal.emit()
            time.sleep(0.2)
        skeletons_copy = self.skeletons[: len(self.skeletons)]
        filename = (Path(self.kinect_mgr.output_dir) / filename).with_suffix('.csv')
        try:
            with open(filename, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                headers, _ = skeletons_copy[0].flatten()
                writer.writerow(headers)
                for sk in skeletons_copy:
                    writer.writerow(sk.flatten()[1])

        except Exception as e:
            logger.error(f"Error dumping skeletons to file {filename}: {e}")
            return
        logger.info(f"Skeletons saved to {filename}. Clearing cache.")
        self.skeletons.clear()
        self.dumpFinishedSignal.emit(str(filename))

    @classmethod
    def build_with_thread(cls, kinect_mgr_cfg: dict) -> 'Tuple[KinectManagerWorker, QThread]':
        worker = cls(kinect_mgr_cfg)
        thread = QThread()
        worker.moveToThread(thread)

        worker.startSkeletonCaptureSignal.connect(worker._on_start)
        worker.pauseSkeletonCaptureSignal.connect(worker._on_pause)
        worker.resumeSkeletonCaptureSignal.connect(worker._on_resume)
        worker.terminateSkeletonCaptureSignal.connect(worker._on_stop)
        worker.dumpSkeletonsSignal.connect(worker._on_dump)

        thread.finished.connect(worker._on_stop)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        return worker, thread