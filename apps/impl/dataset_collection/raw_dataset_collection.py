import sys
import time
import logging
import threading
import csv
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from PySide6.QtCore import Qt, QCoreApplication, QObject, Slot, QMetaObject, QThread, Signal, QTimer
from PySide6.QtWidgets import QApplication

from mwcore.apps import BaseMWOnlineApp
from mwcore.registry import APPS
from mwpose3d.registry import VISUALIZERS
from mwpose3d.utils.typing_utils import ConfigType

from apps.impl.dataset_collection.metadata_input_dialog import InputPopupDialog
from apps.impl.dataset_collection.instruction_thread import InstructionWorker
from apps.impl.dataset_collection.kinect_manager_thread import KinectManagerWorker

# Try to import UdpRawDataReader from mwcore as per user instruction
try:
    from mwcore.radario.readers.TI.DCA1000EVM.udp_raw_reader import UdpRawDataReader
except ImportError:
    # If not available, we define a dummy for type checking or fail later
    UdpRawDataReader = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(threadName)-10s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mwpose3d.visualization.skel_online import OnlineSkeletonVisualizer


class RawRadarWorker(QObject):
    """
    Worker to handle raw radar data capture using UdpRawDataReader.
    Running in a separate thread (via QThread), and spawning a sub-thread for the blocking capture loop.
    """
    # Signals
    started = Signal()
    finished = Signal()
    frame_stats = Signal(str)
    
    # Signal to receive metadata from the UI
    recordMeta = Signal(dict)
    
    # Internal signal to trigger start from controller
    startCaptureSignal = Signal()
    stopCaptureSignal = Signal()

    def __init__(self, reader_cfg: dict):
        super().__init__()
        self.reader_cfg = reader_cfg
        self.reader = None
        self._running = False
        
        # Output directory setup
        self._output_dir = Path(reader_cfg.get('output_dir', './data'))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_file = None
        self._capture_thread: Optional[threading.Thread] = None
        self._metadata_file = self._output_dir / "metadata.csv"

    @Slot(dict)
    def _on_record_meta(self, meta: dict):
        """
        Handle metadata submission. Saves to CSV and prepares filename for next capture.
        """
        file_exists = self._metadata_file.exists()
        try:
            with open(self._metadata_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=meta.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(meta)
            logger.info(f"Metadata recorded to {self._metadata_file}")
        except Exception as e:
            logger.error(f"Failed to write metadata: {e}")

        # Construct filename based on metadata
        # Convention: {ParticipantID}_{Game}_{Description}_{Timestamp}.bin
        pid = meta.get("Participant ID", "P00").replace(" ", "")
        game = meta.get("Game", "Game").replace(" ", "")
        desc = meta.get("Description", "").replace(" ", "")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        filename = f"{pid}_{game}_{timestamp}"
        if desc:
            filename += f"_{desc}"
        filename += ".bin"
        
        self._current_file = self._output_dir / filename
        logger.info(f"Next capture will be saved to: {self._current_file}")

    @Slot()
    def _on_start_capture(self):
        """
        Start the capture process in a separate thread.
        """
        if self._running:
            logger.warning("Capture already running")
            return

        if not self._current_file:
            # Fallback filename if no metadata received
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._current_file = self._output_dir / f"capture_{timestamp}.bin"
            logger.info(f"No metadata received, using fallback filename: {self._current_file}")

        self._capture_thread = threading.Thread(target=self._capture_loop, name="RawRadarCapture", daemon=True)
        self._capture_thread.start()

    def _capture_loop(self):
        """
        Blocking capture loop running in a separate thread.
        """
        logger.info(f"Starting raw radar capture to {self._current_file}")
        
        try:
            # Prepare config for reader
            cfg = self.reader_cfg.copy()
            # Remove output_dir from cfg as it's not a UdpRawDataReader param
            if 'output_dir' in cfg: 
                del cfg['output_dir']
            
            # Initialize reader
            if UdpRawDataReader is None:
                raise ImportError("UdpRawDataReader not available")

            self.reader = UdpRawDataReader(
                save_to_file=str(self._current_file),
                **cfg
            )
            
            self.reader.connect()
            self._running = True
            self.started.emit()
            
            frame_count = 0
            start_time = time.time()
            last_log_time = start_time
            
            logger.info("Radar capture started - waiting for data...")
            
            while self._running:
                # Read frame
                # Note: read() might block, but we expect it to return reasonably often
                data_ok, frame_num, det_obj = self.reader.read()
                
                if data_ok:
                    frame_count += 1
                    num_points = det_obj.get('numObj', 0)
                    
                    # Log stats every second
                    current_time = time.time()
                    if current_time - last_log_time >= 1.0:
                        elapsed = current_time - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        stats_msg = f"Frame: {frame_num} | FPS: {fps:.1f} | Pts: {num_points}"
                        self.frame_stats.emit(stats_msg)
                        last_log_time = current_time
                else:
                    # No data, small sleep
                    time.sleep(0.001)
                    
        except Exception as e:
            logger.error(f"Error in raw radar capture loop: {e}", exc_info=True)
        finally:
            self._cleanup_reader()
            self._running = False
            self.finished.emit()

    def _cleanup_reader(self):
        if self.reader:
            try:
                self.reader.close()
            except Exception as e:
                logger.error(f"Error closing reader: {e}")
            self.reader = None

    @Slot()
    def _on_stop_capture(self):
        """
        Signal to stop the capture loop.
        """
        if not self._running:
            return
            
        logger.info("Stopping raw radar capture...")
        self._running = False
        
        # Wait for thread to join if it's not the current thread
        if self._capture_thread and self._capture_thread.is_alive():
            if threading.current_thread() != self._capture_thread:
                self._capture_thread.join(timeout=2.0)
                if self._capture_thread.is_alive():
                    logger.warning("Capture thread did not finish cleanly")
        
        self._capture_thread = None

    @classmethod
    def build_with_thread(cls, reader_cfg: dict) -> Tuple['RawRadarWorker', QThread]:
        worker = cls(reader_cfg)
        thread = QThread()
        worker.moveToThread(thread)
        
        # Wiring signals to slots within the worker (queued connection by default across threads)
        worker.startCaptureSignal.connect(worker._on_start_capture)
        worker.stopCaptureSignal.connect(worker._on_stop_capture)
        worker.recordMeta.connect(worker._on_record_meta)
        
        # Cleanup
        thread.finished.connect(worker._on_stop_capture)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        
        return worker, thread


@APPS.register_module()
class RawDatasetCollectionApp(BaseMWOnlineApp):
    def __init__(self,
                 reader_cfg: dict,
                 vis_cfg: dict,
                 instructions: dict,
                 kinect_cfg: dict,
                 cfg: ConfigType = None,
                 buffer_cfg: dict = None): # buffer_cfg kept for compatibility but unused
        
        # Initialize base app 
        # Note: BaseMWOnlineApp might create a default reader_thread which we will ignore/override
        # We pass None for reader_cfg to avoid BaseMWOnlineApp trying to build a reader from our raw config
        # which might lack the expected 'type' or point to a non-registered class.
        super().__init__(None, vis_cfg, cfg)
        self.app = QApplication(sys.argv)
        
        # 1. Instruction Worker
        self.instruction_worker, self.instruction_thread = InstructionWorker.build_with_thread(**instructions)
        
        # 2. Kinect Manager Worker
        self.kinect_mgr_worker, self.kinect_mgr_thread = KinectManagerWorker.build_with_thread(**kinect_cfg)
        
        # 3. Raw Radar Worker (Replaces standard reader and buffer)
        # We use the passed reader_cfg for our raw worker
        self.raw_radar_worker, self.raw_radar_thread = RawRadarWorker.build_with_thread(reader_cfg)
        
        # Stop the base class's reader thread if it exists and started (it shouldn't be started yet)
        if hasattr(self, 'reader_thread'):
            # We just ignore it. BaseMWOnlineApp creates it but doesn't start it in __init__ usually.
            pass

        self._controller = None

    def _make_visualizer(self, vis_cfg: dict) -> 'OnlineSkeletonVisualizer':
        visualizer = VISUALIZERS.build(dict(
            vis_cfg,
            on_close=self._on_visualizer_close,
        ))
        return visualizer

    @property
    def visualizer(self) -> 'OnlineSkeletonVisualizer':
        if not hasattr(self, '_visualizer'):
            self._visualizer = self._make_visualizer(self.vis_cfg)
        return self._visualizer
    
    def _on_visualizer_close(self, event):
        if self._controller:
            self._controller.stop_all()
    
    def start(self):
        self._controller = _RawLoopController(self)
        logger.info("Starting Raw Dataset Collection Application")
        self.visualizer.show()
        self._controller.start_all()
        sys.exit(self.app.exec())

    @classmethod
    def from_cfg(cls, cfg: dict):
        return cls(
            reader_cfg=cfg['reader_cfg'],
            vis_cfg=cfg['vis_cfg'],
            instructions=cfg.get('instructions', {}),
            kinect_cfg=cfg['kinect_cfg'],
            cfg=cfg,
            buffer_cfg=cfg.get('buffer_cfg', None)
        )


class _RawLoopController(QObject):
    def __init__(self, app: RawDatasetCollectionApp):
        super().__init__()
        self.app = app

        # Popup for metadata
        self.metainfo_popup = InputPopupDialog(labels=[
            "Participant ID", "Game", "Description"])
        # Connect popup to instruction worker and radar worker
        self.metainfo_popup.accepted.connect(self.app.instruction_worker.instructionsOnInit.emit)
        self.metainfo_popup.rejected.connect(self.app.instruction_worker.instructionsOnInit.emit)
        self.metainfo_popup.submitted.connect(self.app.raw_radar_worker.recordMeta.emit)

        # Wire signals
        
        # 1. Visualizer updates
        # Kinect skeleton -> Visualizer
        self.app.kinect_mgr_worker.recentSkeletonJointCoordSignal.connect(
            self.app.visualizer.update_skeleton, Qt.ConnectionType.QueuedConnection)
            
        # Radar stats -> Visualizer Label
        self.app.raw_radar_worker.frame_stats.connect(
            lambda x: self.app.visualizer.update_label(x))
            
        # 2. Flow Control
        # Init complete -> Check Kinect -> Start Radar
        self.app.instruction_worker.finishedOnInit.connect(
            self._on_init_stage_complete, Qt.ConnectionType.QueuedConnection)
            
        # Stop signal -> Stop cycle
        self.app.instruction_worker.finishedOnStop.connect(
            self._on_cycle_complete, Qt.ConnectionType.QueuedConnection)

    @Slot()
    def before_init(self):
        popup = self.metainfo_popup
        popup.refresh_fields()
        popup.open()

    @Slot()
    def _on_init_stage_complete(self):
        app = self.app
        # Start Kinect if not running
        if not app.kinect_mgr_worker._running:
            app.kinect_mgr_worker.startSkeletonCaptureSignal.emit()

        # Check for Kinect ready
        timer = QTimer(self)
        timer.setInterval(2000)
        
        def _check():
            if app.kinect_mgr_worker._running:
                timer.stop()
                # Start Radar Capture
                app.raw_radar_worker.startCaptureSignal.emit()
                
                # Resume Kinect capture (recording mode)
                app.kinect_mgr_worker.resumeSkeletonCaptureSignal.emit()
                
                # Update Instructions
                app.instruction_worker.instructionsOnStart.emit()
            else:
                logger.info("Waiting for Kinect Manager to start…")
                
        timer.timeout.connect(_check)
        timer.start()

    @Slot()
    def _on_cycle_complete(self):
        logger.info("Cycle complete. Stopping capture.")
        
        # Stop Radar
        self.app.raw_radar_worker.stopCaptureSignal.emit()
        
        # Pause/Dump Kinect
        # We need to trigger dump of skeletons
        # Use a filename derived from metadata or just timestamp
        # Ideally, we should sync this name with the bin file name.
        # But KinectManagerWorker needs a filename.
        # For now, we use a timestamp based name or ask Kinect worker to dump default.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.app.kinect_mgr_worker.dumpSkeletonsSignal.emit(f"kinect_{timestamp}.csv")
        
        # Loop back to init
        self.before_init()

    def start_all(self):
        app = self.app
        # Start threads
        if hasattr(app, 'instruction_thread'): app.instruction_thread.start()
        if hasattr(app, 'kinect_mgr_thread'): app.kinect_mgr_thread.start()
        if hasattr(app, 'raw_radar_thread'): app.raw_radar_thread.start()

        self.before_init()

    def stop_all(self):
        app = self.app
        logger.info("Stopping all threads...")
        
        # Stop Radar
        if hasattr(app, 'raw_radar_worker'):
            app.raw_radar_worker.stopCaptureSignal.emit()
        
        # Stop others
        for tn in (
            "kinect_mgr_thread",
            "instruction_thread",
            "raw_radar_thread",
            # "reader_thread" # Base class thread
        ): 
            if hasattr(app, tn):
                logger.info(f"Requesting interruption for thread: app.{tn}")
                t: "QThread" = getattr(app, tn)
                if t.isRunning():
                    t.requestInterruption()
                    t.quit()
                    t.wait()
