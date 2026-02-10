import argparse
import sys
import os
import time
import logging
import signal
from pathlib import Path
from PySide6.QtCore import QCoreApplication, QObject, QTimer, Slot, Qt, QThread

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(threadName)-10s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("HeadlessRecorder")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from mmengine.config import Config, DictAction
from mwcore.registry import READERS
from mwcore.threads.online_reader import OnlineReaderThread
from apps.impl.dataset_collection.kinect_manager_thread import KinectManagerWorker
# We need PointCloudBufferingWorker for proper dump logic if we want to mimic the app exactly
from apps.impl.dataset_collection.pointcloud_buffer_thread import PointCloudBufferingWorker

class HeadlessRecorder(QObject):
    def __init__(self, cfg, duration_min):
        super().__init__()
        self.cfg = cfg
        self.duration_sec = duration_min * 60
        
        # 1. Setup Reader
        logger.info("Initializing Reader...")
        # Ensure process_point_cloud is False for headless to verify low latency, 
        # OR user might want it True. We respect config.
        # But user said "make it properly dump data".
        self.reader = READERS.build(cfg.reader_cfg)
        self.reader_thread = OnlineReaderThread(self.reader)
        
        # 2. Setup Kinect
        logger.info("Initializing Kinect...")
        self.kinect_worker, self.kinect_thread = KinectManagerWorker.build_with_thread(**cfg.kinect_cfg)
        
        # 3. Setup Buffer (Optional but good for consistency)
        logger.info("Initializing Buffer...")
        self.pcd_buffering_worker, self.pcd_buffering_thread = PointCloudBufferingWorker.build_with_thread(**cfg.buffer_cfg)
        
        # Wire signals roughly like the App
        self.reader_thread.raw_data.connect(
            self.pcd_buffering_worker.enqueue_raw,
            Qt.ConnectionType.QueuedConnection
        )
        self.pcd_buffering_worker.bufferDumped.connect(self.kinect_worker.dumpSkeletonsSignal.emit)
        
        # Capture Control Timer
        self.stop_timer = QTimer(self)
        self.stop_timer.setSingleShot(True)
        self.stop_timer.timeout.connect(self.stop_capture)

        # Connect dump finished to quit
        self.kinect_worker.dumpFinishedSignal.connect(self.on_dump_finished)

    def start(self):
        logger.info("Starting threads...")
        self.reader_thread.start()
        self.kinect_thread.start()
        self.pcd_buffering_thread.start()
        
        # Wait for Kinect to be ready?
        # In the App, there is an init stage. Here we just start.
        # Give a small delay for threads to spin up?
        QTimer.singleShot(2000, self._start_capture)

    def _start_capture(self):
        logger.info("Starting capture...")
        self.kinect_worker.startSkeletonCaptureSignal.emit()
        self.kinect_worker.resumeSkeletonCaptureSignal.emit()
        
        logger.info(f"Recording for {self.duration_sec} seconds...")
        self.stop_timer.start(int(self.duration_sec * 1000))

    @Slot()
    def stop_capture(self):
        logger.info("Duration reached. Stopping capture...")
        
        # Stop Reader
        self.reader_thread.requestInterruption()
        # We manually close reader resources if thread finishes?
        # OnlineReaderThread doesn't have a stop() method, it checks isInterruptionRequested()
        # and has a terminate() which is harsh.
        
        # Dump Logic
        logger.info("Dumping data...")
        
        # 1. Dump Buffer
        self.pcd_buffering_worker.dump_buffer()
        
        # 2. Explicit Dump Kinect (just in case)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.kinect_worker.dumpSkeletonsSignal.emit(f"kinect_{timestamp}.csv")
        
        # Safety timeout to force quit if dump hangs
        QTimer.singleShot(5000, QCoreApplication.instance().quit)

    @Slot(str)
    def on_dump_finished(self, filename):
        logger.info(f"Kinect dump finished: {filename}")
        # We can quit now
        logger.info("Quitting application...")
        QCoreApplication.instance().quit()


def main():
    parser = argparse.ArgumentParser(description="Headless Dataset Collector")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("--duration", type=float, required=True, help="Duration in minutes")
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    app = QCoreApplication(sys.argv)
    
    recorder = HeadlessRecorder(cfg, args.duration)
    recorder.start()
    
    # helper to catch Ctrl+C
    def signal_handler(sig, frame):
        logger.info("Ctrl+C received, stopping...")
        recorder.stop_capture()
        
    signal.signal(signal.SIGINT, signal_handler)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
