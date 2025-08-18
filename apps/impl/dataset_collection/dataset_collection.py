from pathlib import Path
import sys
import time
from typing import Optional, Union, TYPE_CHECKING
import logging

from apps.impl.dataset_collection.metadata_input_dialog import InputPopupDialog
from mwpose3d.utils.typing_utils import ConfigType
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(threadName)-10s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)
from apps.impl.dataset_collection.kinect_manager_thread import KinectManagerWorker
from mwpose3d.utils.pointcloud_toolkits.structures import SimplePointCloud5D

from .pointcloud_buffer_thread import PointCloudBufferingWorker
from .instruction_thread import InstructionWorker
from mwcore.apps import BaseMWOnlineApp
from mwcore.registry import APPS
from mwpose3d.registry import VISUALIZERS
from PySide6.QtCore import Qt, QCoreApplication, QObject, Slot, QMetaObject, QThread, QTimer
QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt


if TYPE_CHECKING:
    from mwpose3d.visualization.skel_online import OnlineSkeletonVisualizer


@APPS.register_module()
class HPEDatasetCollectionApp(BaseMWOnlineApp):
    def __init__(self,
                 reader_cfg: dict,
                 vis_cfg: dict,
                 instructions: dict,
                 buffer_cfg: dict,
                 kinect_cfg: dict,
                 cfg: ConfigType = None):
        super().__init__(reader_cfg, vis_cfg, cfg)
        self.app = QApplication(sys.argv)
        # Thread for displaying text instructions
        self.instruction_worker, self.instruction_thread = InstructionWorker.build_with_thread(**instructions)
        
        # Thread for handling captured point clouds
        self.pcd_buffering_worker, self.pcd_buffering_thread = PointCloudBufferingWorker.build_with_thread(**buffer_cfg)

        # Thread for reading kinect output
        self.kinect_mgr_worker, self.kinect_mgr_thread = KinectManagerWorker.build_with_thread(**kinect_cfg)

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
            cfg=cfg
        )

class _LoopController(QObject):
    def __init__(self, app: HPEDatasetCollectionApp):
        super().__init__()
        self.app = app

        self.metainfo_popup = InputPopupDialog(labels=[
            "Participant ID", "Game", "Position X", "Position Y", "Speed", "Description"])
        self.metainfo_popup.accepted.connect(self.app.instruction_worker.instructionsOnInit.emit)
        self.metainfo_popup.rejected.connect(self.app.instruction_worker.instructionsOnInit.emit)
        self.metainfo_popup.submitted.connect(self.app.pcd_buffering_worker.recordMeta.emit)

        # Wire signals
        self.app.reader_thread.array_data.connect(self.app.visualizer.on_new_cloud, Qt.ConnectionType.QueuedConnection)
        self.app.pcd_buffering_worker.frameCount.connect(lambda x: self.app.visualizer.update_label(f"Frames: {x:04d}"))
        self.app.kinect_mgr_worker.recentSkeletonJointCoordSignal.connect(self.app.visualizer.update_skeleton, Qt.ConnectionType.QueuedConnection)
        self.app.instruction_worker.finishedOnInit.connect(self._on_init_stage_complete, Qt.ConnectionType.QueuedConnection)
        self.app.pcd_buffering_worker.bufferFull.connect(self._on_buffer_full)
        self.app.pcd_buffering_worker.bufferDumped.connect(lambda x: self.app.kinect_mgr_worker.dumpSkeletonsSignal.emit(Path(x).name))
        self.app.instruction_worker.finishedOnStop.connect(self._on_cycle_complete, Qt.ConnectionType.QueuedConnection)

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

        timer = QTimer(self)
        timer.setInterval(2000)
        def _check():
            if app.kinect_mgr_worker._running:
                timer.stop()
                app.reader_thread.raw_data.connect(
                    self.app.pcd_buffering_worker.enqueue_raw,
                    Qt.ConnectionType.QueuedConnection
                )
                app.kinect_mgr_worker.resumeSkeletonCaptureSignal.emit()
                app.instruction_worker.instructionsOnStart.emit()
            else:
                logger.info("Waiting for Kinect Manager to start…")
        timer.timeout.connect(_check)
        timer.start()

    @Slot()
    def _on_buffer_full(self):
        logger.info("Buffer is full. Dumping point cloud data.")
        self.app.pcd_buffering_worker.dump_buffer()
        self.app.reader_thread.raw_data.disconnect(self.app.pcd_buffering_worker.enqueue_raw)
        self.app.instruction_worker.instructionsOnStop.emit(True)
    
    @Slot()
    def _on_cycle_complete(self):
        logger.info("Cycle complete. Preparing for next capture.")
        QMetaObject.invokeMethod(
            self.app.pcd_buffering_worker,
            'clear_buffer',
            Qt.ConnectionType.QueuedConnection,
        )

        self.before_init()

    def start_all(self):
        app = self.app
        for t in (
            app.reader_thread,
            app.kinect_mgr_thread,
            app.instruction_thread,
            app.pcd_buffering_thread,
        ): t.start()

        self.before_init()

    def stop_all(self):
        for tn in (
            "reader_thread",
            "kinect_mgr_thread",
            "instruction_thread",
            "pcd_buffering_thread"
        ): 
            logger.info(f"Requesting interruption for thread: app.{tn}")
            t: "QThread" = getattr(self.app, tn)
            t.requestInterruption()
            logger.info(f"Quitting thread: app.{tn}")
            t.quit()
            logger.info(f"Waiting for thread: app.{tn} to quit")
            t.wait()
