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
        #
        # Phase 1: Start the subprocess in a daemon thread (blocking I/O).
        # Once it's ready, _on_subprocess_ready fires on the QThread event loop.
        self._startup_thread = threading.Thread(
            target=self._start_subprocess, name="KinectStartup", daemon=True)
        self._startup_thread.start()

    def _start_subprocess(self):
        """Blocking helper – runs in a one-shot daemon thread."""
        try:
            logger.info("[Kinect] Starting Kinect subprocess...")
            self.kinect_mgr.start_skeleton_capture()
            logger.info("[Kinect] Waiting for capture to start (checking for mmap data)...")
            self.kinect_mgr.wait_for_capture_starts()
            logger.info("[Kinect] Capture process started successfully!")
        except Exception as e:
            logger.error(f"[Kinect] Error starting Kinect capture: {e}")
            return

        # Start daemon threads to log subprocess stdout/stderr
        if self.kinect_mgr.process:
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

            threading.Thread(target=log_subprocess_output, daemon=True).start()
            threading.Thread(target=log_subprocess_error, daemon=True).start()

        # Transition to Phase 2 on the QThread event loop
        from PySide6.QtCore import QTimer, QMetaObject, Qt
        QMetaObject.invokeMethod(self, '_on_subprocess_ready', Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _on_subprocess_ready(self):
        """Phase 2: subprocess is up – start QTimer-based skeleton polling."""
        self._running = self.kinect_mgr.running
        self._paused = False
        self._skeleton_count = 0
        self.captureProcessStartedSignal.emit()

        from PySide6.QtCore import QTimer
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(10)  # 10ms poll interval
        self._poll_timer.timeout.connect(self._poll_skeletons)
        self._poll_timer.start()
        logger.info("[Kinect] Skeleton polling started (QTimer, 10ms).")

    @Slot()
    def _poll_skeletons(self):
        """Called every 10ms by QTimer – reads new skeletons from mmap."""
        if not self._running or QThread.currentThread().isInterruptionRequested():
            self._poll_timer.stop()
            logger.info(f"[Kinect] Polling stopped. Total skeletons: {self._skeleton_count}")
            return
        if self._paused:
            return

        try:
            new = self.kinect_mgr.get_new_records()
        except Exception as e:
            logger.error(f"[Kinect] Error getting new records: {e}")
            self._poll_timer.stop()
            return

        if new:
            self.skeletons.extend(new)
            self._skeleton_count += len(new)
            if self._skeleton_count == 1:
                logger.info("[Kinect] First skeleton frame received!")
            elif self._skeleton_count % 100 == 0:
                logger.info(f"[Kinect] Captured {self._skeleton_count} skeleton frames so far")
            self.recentSkeletonJointCoordSignal.emit(
                self.skeletons[-1].flatten()[1][2:]
            )

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
        if hasattr(self, '_poll_timer') and self._poll_timer.isActive():
            self._poll_timer.stop()
        if hasattr(self, '_startup_thread') and self._startup_thread.is_alive():
            logger.info("Waiting for Kinect startup thread to finish...")
            self._startup_thread.join(timeout=1)
        self.kinect_mgr.refresh()
        self._running = False
        self._paused = False
        logger.info("Kinect Capture stopped.")

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