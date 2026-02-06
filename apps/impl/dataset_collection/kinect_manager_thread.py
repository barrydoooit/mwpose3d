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
        # Don't call refresh() here - it closes/recreates the mmap which conflicts
        # with the subprocess trying to create the same named mmap
        # refresh() should only be called when STOPPING to clean up stale state
        self._ct = threading.Thread(target=self._run_capture_loop, name="KinectCaptureLoop", daemon=True)
        self._ct.start()
    
    def _run_capture_loop(self):
        try:
            logger.info("[Kinect] Starting Kinect subprocess...")
            self.kinect_mgr.start_skeleton_capture()
            logger.info("[Kinect] Waiting for capture to start (checking for mmap data)...")
            self.kinect_mgr.wait_for_capture_starts()
            self.captureProcessStartedSignal.emit()
            logger.info("[Kinect] Capture process started successfully!")
        except Exception as e:
            logger.error(f"[Kinect] Error starting Kinect capture: {e}")
            return

        self._running = self.kinect_mgr.running
        self._paused = False
        
        # Start thread to monitor subprocess output
        if self.kinect_mgr.process:
            import threading
            def log_subprocess_output():
                if self.kinect_mgr.process.stdout:
                    for line in iter(self.kinect_mgr.process.stdout.readline, b''):
                        if line:
                            logger.info(f"[Kinect stdout] {line.decode().strip()}")
            
            def log_subprocess_error():
                if self.kinect_mgr.process.stderr:
                    for line in iter(self.kinect_mgr.process.stderr.readline, b''):
                        if line:
                            logger.warning(f"[Kinect stderr] {line.decode().strip()}")
            
            stdout_thread = threading.Thread(target=log_subprocess_output, daemon=True)
            stderr_thread = threading.Thread(target=log_subprocess_error, daemon=True)
            stdout_thread.start()
            stderr_thread.start()
        
        logger.info("[Kinect] Entering skeleton capture loop...")
        skeleton_count = 0
        while self._running and not QThread.currentThread().isInterruptionRequested():
            if self._paused:
                time.sleep(0.1)
                continue
            try:
                new = self.kinect_mgr.get_new_records()
            except Exception as e:
                logger.error(f"[Kinect] Error getting new records: {e}")
                break

            if new:
                self.skeletons.extend(new)
                skeleton_count += len(new)
                if skeleton_count == 1:
                    logger.info("[Kinect] First skeleton frame received!")
                elif skeleton_count % 100 == 0:
                    logger.info(f"[Kinect] Captured {skeleton_count} skeleton frames so far")
                self.recentSkeletonJointCoordSignal.emit(self.skeletons[-1].flatten()[1][2:])  # Emit only the flattened data excluding timestamp and unix_ms
            
            time.sleep(0.01)
        
        logger.info(f"[Kinect] Capture loop ended. Total skeletons captured: {skeleton_count}")

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