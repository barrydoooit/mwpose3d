from pathlib import Path
import sys
import time
from typing import Literal, Optional, Union, TYPE_CHECKING
import logging

from mmengine.config import Config

from apps.impl.online_estim_and_collection.key_event_collector import KeyEventTraceWorker

from .source_selection_dialog import SourceSelectDialog
from ..dataset_collection.metadata_input_dialog import InputPopupDialog
from ..online_skeleton_estim.estimation_worker import InferenceWorker, InferenceWorkerThread
from mwpose3d.utils.typing_utils import ConfigType
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(threadName)-10s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)
from ..dataset_collection.kinect_manager_thread import KinectManagerWorker
from mwpose3d.utils.pointcloud_toolkits.structures import SimplePointCloud5D

from ..dataset_collection.pointcloud_buffer_thread import PointCloudBufferingWorker
from ..dataset_collection.instruction_thread import InstructionWorker
from mwcore.apps import BaseMWOnlineApp
from mwcore.registry import APPS
from mwpose3d.registry import VISUALIZERS
from PySide6.QtCore import Qt, QCoreApplication, QObject, Slot, QMetaObject, QThread, QTimer, QEventLoop
QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


if TYPE_CHECKING:
    from mwpose3d.visualization.skel_online import OnlineSkeletonVisualizer


@APPS.register_module()
class HPESimulCollectionPredictionApp(BaseMWOnlineApp):
    def __init__(self,
                 reader_cfg: dict,
                 vis_cfg: dict,
                 instructions: dict,
                 buffer_cfg: dict,
                 kinect_cfg: dict,
                 hpe_model_cfg: Union[Path, str, dict],
                 inference_worker_cfg: dict = {},
                 load_from: Optional[Union[Path, str]] = None,
                 key_event_collector_cfg: Optional[dict] = None,
                 cfg: ConfigType = None):
        super().__init__(reader_cfg, vis_cfg, cfg)
        self.app = QApplication(sys.argv)

        self.instruction_worker, self.instruction_thread = InstructionWorker.build_with_thread(**instructions)
        self.pcd_buffering_worker, self.pcd_buffering_thread = PointCloudBufferingWorker.build_with_thread(**buffer_cfg)
        self.kinect_mgr_worker, self.kinect_mgr_thread = KinectManagerWorker.build_with_thread(**kinect_cfg)
        self.inference_worker, self.inference_thread = InferenceWorker.build_with_thread(self, inference_worker_cfg)

        self.key_event_worker = None if key_event_collector_cfg is None else KeyEventTraceWorker(**key_event_collector_cfg)

        self.hpe_model_cfg = self._load_hpe_cfg(hpe_model_cfg, load_from)
        self._use_prediction: bool = True
        self._selected_source: Literal["mmwave", "kinect"] = "mmwave"
        self._controller = None
    
    @staticmethod
    def _load_hpe_cfg(hpe_model_cfg: Union[Path, str, dict], load_from: Optional[str]) -> dict:
        cfg = hpe_model_cfg if isinstance(hpe_model_cfg, dict) else Config.fromfile(hpe_model_cfg)
        if load_from is not None:
            cfg.load_from = load_from
            logger.info(f"Loading checkpoint from {load_from}")
        return cfg

    @property
    def inference_engine(self):
        if not hasattr(self, '_inference_engine'):
            from mwpose3d.runner.inference_engine import InferenceEngine
            self._inference_engine = InferenceEngine.from_cfg(self.hpe_model_cfg)
            logger.info(f"Loaded inference engine with model: {self._inference_engine.model.__class__.__name__}")
        return self._inference_engine

    # @property
    # def inference_thread(self) -> InferenceWorkerThread:
    #     if not hasattr(self, '_inference_thread'):
    #         self._inference_thread = InferenceWorkerThread(self, parent=None)
    #         logger.info("Inference worker thread initialized.")
    #     return self._inference_thread
    
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
        self._controller.stop_all()
    
    def start(self):
        self._controller = _LoopController(self)
        logger.info("Starting HPE Dataset Collection Application")
        self.visualizer.show()
        self._controller.start_all()
        sys.exit(self.app.exec())

    @classmethod
    def from_cfg(cls, cfg: dict):
        return cls(
            reader_cfg=cfg['reader_cfg'],
            vis_cfg=cfg['vis_cfg'],
            instructions=cfg.get('instructions', {}),
            buffer_cfg  =cfg['buffer_cfg'], 
            kinect_cfg=cfg['kinect_cfg'],
            hpe_model_cfg=cfg['hpe_model_cfg'],
            inference_worker_cfg=cfg.get('inference_worker_cfg', {}),
            load_from=cfg['load_from'],
            key_event_collector_cfg=cfg.get('key_event_collector_cfg', None),
            cfg=cfg
        )

class _LoopController(QObject):
    def __init__(self, app: HPESimulCollectionPredictionApp):
        super().__init__()
        self.app = app

        self.metainfo_popup = InputPopupDialog(labels=[
            "Participant ID", "Game", "Position X", "Position Y", "Speed", "Description"])
        self.metainfo_popup.submitted.connect(self.app.pcd_buffering_worker.recordMeta.emit)
        self.metainfo_popup.finished.connect(self._on_metainfo_closed, Qt.ConnectionType.QueuedConnection)

        # Wire signals
        self.app.reader_thread.array_data.connect(self.app.visualizer.on_new_cloud, Qt.ConnectionType.QueuedConnection)
        self.app.pcd_buffering_worker.frameCount.connect(lambda x: self.app.visualizer.update_label(f"Frames: {x:04d}"))
        self.app.pcd_buffering_worker.bufferDumped.connect(self.app.kinect_mgr_worker.dumpSkeletonsSignal.emit)
        if self.app.key_event_worker is not None:
            self.app.pcd_buffering_worker.bufferDumped.connect(self.app.key_event_worker.dump_to_json, Qt.ConnectionType.QueuedConnection)

        self.app.instruction_worker.finishedOnInit.connect(self._on_init_stage_complete, Qt.ConnectionType.QueuedConnection)
        self.app.pcd_buffering_worker.bufferFull.connect(self._on_buffer_full)
        self.app.instruction_worker.finishedOnStop.connect(self._on_cycle_complete, Qt.ConnectionType.QueuedConnection)
    
    @Slot(int)
    def _on_metainfo_closed(self, _result_code: int):
        dlg = SourceSelectDialog(parent=self.app.visualizer)
        dlg.exec()
        self.app._selected_source = dlg.selected or "kinect"
        self.app._use_prediction = (self.app._selected_source == "mmwave")

        try:
            if self.app._use_prediction:
                self.app.kinect_mgr_worker.kinect_mgr.mode = ['capture']
            else:
                self.app.kinect_mgr_worker.kinect_mgr.mode = ['capture', 'control']
        except Exception as e:
            logger.error(f"Error setting Kinect mode: {e}")
        self.app.instruction_worker.instructionsOnInit.emit()

    @Slot()
    def before_init(self):
        popup = self.metainfo_popup
        popup.refresh_fields()
        popup.open()

    @Slot()
    def _on_init_stage_complete(self):
        app = self.app
        if not app.kinect_mgr_worker._running:
            app.kinect_mgr_worker.startSkeletonCaptureSignal.emit()
        try: app.kinect_mgr_worker.recentSkeletonJointCoordSignal.disconnect(app.visualizer.update_skeleton)
        except Exception: pass
        try: app.inference_worker.inference_done.disconnect(app.visualizer.update_skeleton)
        except Exception: pass
        try: app.reader_thread.array_data.disconnect(app.inference_worker.enqueue)
        except Exception: pass

        if app._use_prediction:
            app.reader_thread.array_data.connect(app.inference_worker.enqueue, Qt.ConnectionType.QueuedConnection)
            app.inference_worker.inference_done.connect(app.visualizer.update_skeleton, Qt.ConnectionType.QueuedConnection)
        else:
            print("Connecting Kinect skeleton to visualizer directly.")
            app.kinect_mgr_worker.recentSkeletonJointCoordSignal.connect(app.visualizer.update_skeleton, Qt.ConnectionType.QueuedConnection)
        started_loop = QEventLoop()
        started = {'ok': False}
        def _on_started():
            started['ok'] = True
            logger.info("Kinect Manager started.")
            started_loop.quit()
        try:
            app.kinect_mgr_worker.captureProcessStartedSignal.connect(_on_started, Qt.ConnectionType.QueuedConnection)
        except Exception:
            QTimer.singleShot(1000, started_loop.quit)

        QTimer.singleShot(3000, started_loop.quit)
        started_loop.exec()
        logger.info("Waiting for Kinect Manager to start…") # TODO: check if this is printed after Kinect Manager Started message
        try: app.reader_thread.raw_data.disconnect(app.pcd_buffering_worker.enqueue_raw)
        except Exception: pass
        app.reader_thread.raw_data.connect(app.pcd_buffering_worker.enqueue_raw, Qt.ConnectionType.QueuedConnection)

        app.kinect_mgr_worker.resumeSkeletonCaptureSignal.emit()
        app.instruction_worker.instructionsOnStart.emit()

    @Slot()
    def _on_buffer_full(self):
        logger.info("Buffer is full. Dumping point cloud data.")
        self.app.pcd_buffering_worker.dump_buffer()
        try:
            self.app.reader_thread.raw_data.disconnect(self.app.pcd_buffering_worker.enqueue_raw)
        except Exception: pass
        self.app.instruction_worker.instructionsOnStop.emit(True)
    
    @Slot()
    def _on_cycle_complete(self):
        logger.info("Cycle complete. Preparing for next capture.")
        QMetaObject.invokeMethod(
            self.app.pcd_buffering_worker,
            'clear_buffer',
            Qt.ConnectionType.QueuedConnection,
        )

        # Wait for Kinect process to stop
        loop = QEventLoop()
        timer = QTimer()
        timer.setInterval(100)
        timer.timeout.connect(lambda: None)
        def check_stopped():
            try:
                proc = self.app.kinect_mgr_worker.kinect_mgr.process
            except Exception:
                proc = None
            if proc is None:
                loop.quit()
        timer.timeout.connect(check_stopped)
        timer.start()
        self.app.kinect_mgr_worker.terminateSkeletonCaptureSignal.emit()
        QTimer.singleShot(8000, loop.quit)
        loop.exec()
        timer.stop()
        logger.info("Kinect capture program confirmed stopped. Clearing buffers.")

        self.before_init()

    def start_all(self):
        app = self.app
        for t in (
            app.reader_thread,
            app.kinect_mgr_thread,
            app.instruction_thread,
            app.pcd_buffering_thread,
            app.inference_thread
        ): t.start()

        self.before_init()

    def stop_all(self):
        if self.app.key_event_worker is not None:
            self.app.key_event_worker.stop()
        try:
            self.app.kinect_mgr_worker.terminateSkeletonCaptureSignal.emit()
        except Exception: pass
        for tn in (
            "reader_thread",
            "kinect_mgr_thread",
            "instruction_thread",
            "pcd_buffering_thread",
            "inference_thread"
        ):
            logger.info(f"Requesting interruption for thread: app.{tn}")
            t: "QThread" = getattr(self.app, tn)
            t.requestInterruption()
            logger.info(f"Quitting thread: app.{tn}")
            t.quit()
            logger.info(f"Waiting for thread: app.{tn} to quit")
            t.wait()
