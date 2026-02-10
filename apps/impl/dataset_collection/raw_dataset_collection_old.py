# import sys
# import time
# import logging
# import threading
# import csv
# from pathlib import Path
# from typing import TYPE_CHECKING, Optional, Tuple

# from PySide6.QtCore import Qt, QCoreApplication, QObject, Slot, QMetaObject, QThread, Signal, QTimer
# QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
# from PySide6.QtWidgets import QApplication

# from mwcore.apps import BaseMWOnlineApp
# from mwcore.registry import APPS
# from mwpose3d.registry import VISUALIZERS
# from mwpose3d.utils.typing_utils import ConfigType

# from apps.impl.dataset_collection.metadata_input_dialog import InputPopupDialog
# from apps.impl.dataset_collection.instruction_thread import InstructionWorker
# from apps.impl.dataset_collection.kinect_manager_thread import KinectManagerWorker


# # Try to import UdpRawDataReader from mwcore as per user instruction
# # REMOVED: In-process UdpRawDataReader import as we use external script now
# import subprocess

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(threadName)-10s %(levelname)-8s %(name)s: %(message)s",
#     stream=sys.stdout,
#     force=True
# )
# logger = logging.getLogger(__name__)

# if TYPE_CHECKING:
#     from mwpose3d.visualization.skel_online import OnlineSkeletonVisualizer



# class MwCoreRawRadarWorker(QObject):
#     """
#     Worker to handle raw radar data capture by executing an external script via uv run.
#     Replaces the previous in-process UdpRawDataReader.
#     """
#     # Signals
#     started = Signal()
#     finished = Signal()
#     frame_stats = Signal(str)
    
#     # Signal to receive metadata from the UI
#     recordMeta = Signal(dict)
    
#     # Internal signals
#     startCaptureSignal = Signal()
#     stopCaptureSignal = Signal()

#     def __init__(self, reader_cfg: dict):
#         super().__init__()
#         self.reader_cfg = reader_cfg
#         self._process = None
#         self._running = False
        
#         # Output directory setup
#         self._output_dir = Path(reader_cfg.get('output_dir', './data'))
#         self._output_dir.mkdir(parents=True, exist_ok=True)
        
#         self._current_file = None
#         self._metadata_file = self._output_dir / "metadata.csv"
        
#         # Path to the external script
#         # Assuming fixed location based on workspace structure
#         self._script_path = Path("e:/Projects/mwCore/tools/capture_raw_data.py")

#     @Slot(dict)
#     def _on_record_meta(self, meta: dict):
#         """
#         Handle metadata submission. Saves to CSV and prepares filename for next capture.
#         """
#         file_exists = self._metadata_file.exists()
#         try:
#             with open(self._metadata_file, 'a', newline='') as f:
#                 writer = csv.DictWriter(f, fieldnames=meta.keys())
#                 if not file_exists:
#                     writer.writeheader()
#                 writer.writerow(meta)
#             logger.info(f"Metadata recorded to {self._metadata_file}")
#         except Exception as e:
#             logger.error(f"Failed to write metadata: {e}")

#         # Construct filename based on metadata
#         pid = meta.get("Participant ID", "P00").replace(" ", "")
#         game = meta.get("Game", "Game").replace(" ", "")
#         desc = meta.get("Description", "").replace(" ", "")
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
        
#         filename = f"{pid}_{game}_{timestamp}"
#         if desc:
#             filename += f"_{desc}"
#         filename += ".bin"
        
#         self._current_file = self._output_dir / filename
#         logger.info(f"Next capture will be saved to: {self._current_file}")

#     @Slot()
#     def _on_start_capture(self):
#         if self._running:
#             logger.warning("Capture already running")
#             return

#         if not self._current_file:
#             timestamp = time.strftime("%Y%m%d_%H%M%S")
#             self._current_file = self._output_dir / f"capture_{timestamp}.bin"
#             logger.info(f"No metadata received, using fallback filename: {self._current_file}")
            
#         logger.info(f"Starting external radar capture script: {self._script_path}")
#         logger.info(f"Saving to: {self._current_file}")
        
#         try:
#             # Construct command: uv run <script> --save-file <file>
#             # Note: We don't pass other config params as arguments since the script 
#             # likely uses default or internal config for the reader. 
#             # If specific params are needed, they should be added here.
#             cmd = [
#                 "uv", "run",
#                 str(self._script_path),
#                 "--save-file", str(self._current_file)
#             ]
            
#             # Start subprocess
#             # We use bufsize=1 for line buffering
#             # Start subprocess
#             # Use sys.executable to run directly, avoiding uv overhead/signals
#             cmd = [
#                 sys.executable,
#                 str(self._script_path),
#                 "--save-file", str(self._current_file)
#             ]
            
#             logger.info(f"Running command: {cmd}")
            
#             self._process = subprocess.Popen(
#                 cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE
#                 # Default buffering, binary mode
#             )
            
#             self._running = True
#             self.started.emit()
            
#             # Start threads to monitor stdout/stderr without blocking
#             self._start_log_threads()
            
#         except Exception as e:
#             logger.error(f"Failed to start external capture script: {e}")
#             self._running = False
#             self.finished.emit()

#     def _start_log_threads(self):
#         """Starts threads to read stdout and stderr from the subprocess."""
        
#         def log_stdout():
#             if not self._process or not self._process.stdout:
#                 return
#             try:
#                 # Read binary lines
#                 for line_bytes in iter(self._process.stdout.readline, b''):
#                     if line_bytes:
#                         try:
#                             line = line_bytes.decode('utf-8', errors='replace').strip()
#                         except:
#                             continue
                            
#                         # Filtering: Reduce log spam to avoid blocking the pipe/GUI
#                         # Only log frame stats and errors/warnings
#                         is_stats = "Frame" in line and "|" in line
                        
#                         if is_stats:
#                             logger.info(f"[RadarScript] {line}")
#                             self.frame_stats.emit(line)
#                         else:
#                             # Log other output as debug, or if it looks important
#                             if "Error" in line or "Warning" in line or "Exception" in line:
#                                 logger.warning(f"[RadarScript Out] {line}")
#                             else:
#                                 logger.debug(f"[RadarScript Debug] {line}")
#             except Exception as e:
#                 logger.error(f"Error reading stdout: {e}")

#         def log_stderr():
#             if not self._process or not self._process.stderr:
#                 return
#             try:
#                 for line_bytes in iter(self._process.stderr.readline, b''):
#                     if line_bytes:
#                         try:
#                             line = line_bytes.decode('utf-8', errors='replace').strip()
#                             logger.warning(f"[RadarScript Error] {line}")
#                         except:
#                             pass
#             except Exception as e:
#                 logger.error(f"Error reading stderr: {e}")

#         t_out = threading.Thread(target=log_stdout, daemon=True, name="RadarScriptStdout")
#         t_err = threading.Thread(target=log_stderr, daemon=True, name="RadarScriptStderr")
#         t_out.start()
#         t_err.start()

#     @Slot()
#     def _on_stop_capture(self):
#         if not self._running:
#             return
            
#         logger.info("Stopping external radar capture...")
        
#         if self._process:
#             # Try to terminate gracefully
#             self._process.terminate()
#             try:
#                 self._process.wait(timeout=2.0)
#             except subprocess.TimeoutExpired:
#                 logger.warning("Capture script did not exit in time, killing...")
#                 self._process.kill()
            
#             logger.info("External capture script stopped.")
#             self._process = None
            
#         self._running = False
#         self.finished.emit()

#     @classmethod
#     def build_with_thread(cls, reader_cfg: dict) -> Tuple['MwCoreRawRadarWorker', QThread]:
#         worker = cls(reader_cfg)
#         thread = QThread()
#         worker.moveToThread(thread)
        
#         # Wiring signals/slots
#         worker.startCaptureSignal.connect(worker._on_start_capture)
#         worker.stopCaptureSignal.connect(worker._on_stop_capture)
#         worker.recordMeta.connect(worker._on_record_meta)
        
#         # Cleanup
#         thread.finished.connect(worker._on_stop_capture)
#         thread.finished.connect(worker.deleteLater)
#         thread.finished.connect(thread.deleteLater)
        
#         return worker, thread



# @APPS.register_module()
# class RawDatasetCollectionApp(BaseMWOnlineApp):
#     def __init__(self,
#                  reader_cfg: dict,
#                  vis_cfg: dict,
#                  instructions: dict,
#                  kinect_cfg: dict,
#                  cfg: ConfigType = None,
#                  buffer_cfg: dict = None): # buffer_cfg kept for compatibility but unused
        
#         # Initialize base app 
#         # Note: BaseMWOnlineApp might create a default reader_thread which we will ignore/override
#         # We pass None for reader_cfg to avoid BaseMWOnlineApp trying to build a reader from our raw config
#         # which might lack the expected 'type' or point to a non-registered class.
#         super().__init__(None, vis_cfg, cfg)
#         self.app = QApplication(sys.argv)
        
#         # 1. Instruction Worker
#         self.instruction_worker, self.instruction_thread = InstructionWorker.build_with_thread(**instructions)
        
#         # 2. Kinect Manager Worker
#         self.kinect_mgr_worker, self.kinect_mgr_thread = KinectManagerWorker.build_with_thread(**kinect_cfg)
        
#         # 3. Raw Radar Worker (Replaces standard reader and buffer)
#         # We use the passed reader_cfg for our raw worker
#         self.raw_radar_worker, self.raw_radar_thread = MwCoreRawRadarWorker.build_with_thread(reader_cfg)
        
#         # Stop the base class's reader thread if it exists and started (it shouldn't be started yet)
#         if hasattr(self, 'reader_thread'):
#             # We just ignore it. BaseMWOnlineApp creates it but doesn't start it in __init__ usually.
#             pass

#         self._controller = None

#     def _make_visualizer(self, vis_cfg: dict) -> 'OnlineSkeletonVisualizer':
#         visualizer = VISUALIZERS.build(dict(
#             vis_cfg,
#             on_close=self._on_visualizer_close,
#         ))
#         return visualizer

#     @property
#     def visualizer(self) -> 'OnlineSkeletonVisualizer':
#         if not hasattr(self, '_visualizer'):
#             self._visualizer = self._make_visualizer(self.vis_cfg)
#         return self._visualizer
    
#     def _on_visualizer_close(self, event):
#         if self._controller:
#             self._controller.stop_all()
    
#     def start(self):
#         self._controller = _RawLoopController(self)
#         logger.info("Starting Raw Dataset Collection Application")
#         self.visualizer.show()
#         self._controller.start_all()
#         sys.exit(self.app.exec())

#     @classmethod
#     def from_cfg(cls, cfg: dict):
#         return cls(
#             reader_cfg=cfg['reader_cfg'],
#             vis_cfg=cfg['vis_cfg'],
#             instructions=cfg.get('instructions', {}),
#             kinect_cfg=cfg['kinect_cfg'],
#             cfg=cfg,
#             buffer_cfg=cfg.get('buffer_cfg', None)
#         )


# class _RawLoopController(QObject):
#     def __init__(self, app: RawDatasetCollectionApp):
#         super().__init__()
#         self.app = app

#         # Popup for metadata
#         self.metainfo_popup = InputPopupDialog(labels=[
#             "Participant ID", "Game", "Description"])
#         # Connect popup to instruction worker and radar worker
#         self.metainfo_popup.accepted.connect(self.app.instruction_worker.instructionsOnInit.emit)
#         self.metainfo_popup.rejected.connect(self.app.instruction_worker.instructionsOnInit.emit)
#         self.metainfo_popup.submitted.connect(self.app.raw_radar_worker.recordMeta.emit)

#         # Wire signals
        
#         # 1. Visualizer updates
#         # Kinect skeleton -> Visualizer
#         self.app.kinect_mgr_worker.recentSkeletonJointCoordSignal.connect(
#             self.app.visualizer.update_skeleton, Qt.ConnectionType.QueuedConnection)
            
#         # Radar stats -> Visualizer Label
#         self.app.raw_radar_worker.frame_stats.connect(
#             lambda x: self.app.visualizer.update_label(x))
            
#         # 2. Flow Control
#         # Kinect startup signal -> Start radar and instructions
#         self.app.kinect_mgr_worker.captureProcessStartedSignal.connect(
#             self._on_kinect_started, Qt.ConnectionType.QueuedConnection)
        
#         # Init complete -> Start Kinect
#         self.app.instruction_worker.finishedOnInit.connect(
#             self._on_init_stage_complete, Qt.ConnectionType.QueuedConnection)
            
#         # Stop signal -> Stop cycle
#         self.app.instruction_worker.finishedOnStop.connect(
#             self._on_cycle_complete, Qt.ConnectionType.QueuedConnection)

#     @Slot()
#     def before_init(self):
#         popup = self.metainfo_popup
#         popup.refresh_fields()
#         popup.open()

#     @Slot()
#     def _on_init_stage_complete(self):
#         app = self.app
#         if not app.kinect_mgr_worker._running:
#             logger.info("Starting Kinect Manager...")
#             app.kinect_mgr_worker.startSkeletonCaptureSignal.emit()
#         else:
#             # Already running, just trigger the started handler
#             self._on_kinect_started()
    
#     @Slot()
#     def _on_kinect_started(self):
#         """Called when Kinect process has successfully started."""
#         logger.info("Kinect Manager started successfully.")
#         app = self.app
        
#         # Start Radar Capture
#         app.raw_radar_worker.startCaptureSignal.emit()
        
#         # Resume Kinect capture (recording mode)
#         app.kinect_mgr_worker.resumeSkeletonCaptureSignal.emit()
        
#         # Update Instructions
#         app.instruction_worker.instructionsOnStart.emit()

#     @Slot()
#     def _on_cycle_complete(self):
#         logger.info("Cycle complete. Stopping capture.")
        
#         # Stop Radar
#         self.app.raw_radar_worker.stopCaptureSignal.emit()
        
#         # Pause/Dump Kinect
#         # We need to trigger dump of skeletons
#         # Use a filename derived from metadata or just timestamp
#         # Ideally, we should sync this name with the bin file name.
#         # But KinectManagerWorker needs a filename.
#         # For now, we use a timestamp based name or ask Kinect worker to dump default.
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         self.app.kinect_mgr_worker.dumpSkeletonsSignal.emit(f"kinect_{timestamp}.csv")
        
#         # Loop back to init
#         self.before_init()

#     def start_all(self):
#         app = self.app
#         # Start threads
#         if hasattr(app, 'instruction_thread'): app.instruction_thread.start()
#         if hasattr(app, 'kinect_mgr_thread'): app.kinect_mgr_thread.start()
#         if hasattr(app, 'raw_radar_thread'): app.raw_radar_thread.start()

#         self.before_init()

#     def stop_all(self):
#         app = self.app
#         logger.info("Stopping all threads...")
        
#         # Stop Radar
#         if hasattr(app, 'raw_radar_worker'):
#             app.raw_radar_worker.stopCaptureSignal.emit()
        
#         # Stop others
#         for tn in (
#             "kinect_mgr_thread",
#             "instruction_thread",
#             "raw_radar_thread",
#             # "reader_thread" # Base class thread
#         ): 
#             if hasattr(app, tn):
#                 logger.info(f"Requesting interruption for thread: app.{tn}")
#                 t: "QThread" = getattr(app, tn)
#                 if t.isRunning():
#                     t.requestInterruption()
#                     t.quit()
#                     t.wait()
