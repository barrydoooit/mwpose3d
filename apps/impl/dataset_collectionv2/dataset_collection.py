import logging
import sys
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QCoreApplication, QObject, QMetaObject, QThread, QTimer, Qt, Slot
from PySide6.QtWidgets import QApplication

from apps.impl.dataset_collection.instruction_thread import InstructionWorker
from apps.impl.dataset_collection.metadata_input_dialog import InputPopupDialog
from .buffer_thread import PointCloudBufferingWorker
from .kinect_manager_thread import KinectManagerWorker
from mwcore.apps import BaseMWApp
from mwcore.registry import APPS
from mwpose3d.registry import VISUALIZERS
from mwpose3d.utils.typing_utils import ConfigType

QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(threadName)-10s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mwpose3d.visualization.skel_online import OnlineSkeletonVisualizer


@APPS.register_module()
class HPEDatasetCollectionAppV2(BaseMWApp):
    _CONNECTION_ALIASES = {
        "reader.array_data": "reader_worker.array_data",
        "reader.raw_data": "reader_worker.raw_data",
        "reader.frame_signal": "reader_worker.frame_signal",
        "reader_thread.array_data": "reader_thread.array_data",
        "reader_thread.raw_data": "reader_thread.raw_data",
        "reader_thread.frame_signal": "reader_thread.frame_signal",
        "popup.accepted": "controller.metainfo_popup.accepted",
        "popup.rejected": "controller.metainfo_popup.rejected",
        "popup.submitted": "controller.metainfo_popup.submitted",
    }

    def __init__(
        self,
        thread_cfg: dict,
        vis_cfg: dict,
        instructions: dict,
        buffer_cfg: dict,
        kinect_cfg: dict,
        loop_cfg: dict,
        connections: Optional[list[dict]] = None,
        cfg: ConfigType = None,
    ):
        super().__init__(thread_cfg=thread_cfg, vis_cfg=vis_cfg, connections=connections, cfg=cfg)

        self.app = QApplication.instance() or QApplication(sys.argv)

        # Thread for displaying text instructions
        self.instruction_worker, self.instruction_thread = InstructionWorker.build_with_thread(**instructions)

        # Thread for handling captured point clouds
        self.pcd_buffering_worker, self.pcd_buffering_thread = PointCloudBufferingWorker.build_with_thread(
            **buffer_cfg
        )

        # Thread for reading kinect output
        self.kinect_mgr_worker, self.kinect_mgr_thread = KinectManagerWorker.build_with_thread(**kinect_cfg)

        self.loop_cfg = loop_cfg
        self._controller = None

    @property
    def controller(self) -> "_LoopController":
        if self._controller is None:
            raise RuntimeError("Controller is not created yet. Call start() first.")
        return self._controller

    @classmethod
    def _alias_path(cls, path: str) -> str:
        return cls._CONNECTION_ALIASES.get(path, path)

    @classmethod
    def _parse_collection_connections(cls, connections: list[dict]) -> list[dict]:
        parsed: list[dict] = []
        for idx, conn in enumerate(connections):
            if not isinstance(conn, dict):
                raise TypeError(
                    f"Invalid connection at index {idx}: expected dict, got {type(conn)}"
                )

            signal = conn.get("signal", conn.get("on"))
            slot = conn.get("slot", conn.get("to"))
            if signal is None or slot is None:
                raise ValueError(
                    f"Invalid connection at index {idx}: requires signal/slot "
                    f"(or on/to), got {conn}"
                )

            signal_path = cls._alias_path(signal)
            slot_path = cls._alias_path(slot)
            conn_type = conn.get("type", conn.get("connection_type", "AutoConnection"))

            parsed.append(dict(signal=signal_path, slot=slot_path, type=conn_type))

        return parsed

    @classmethod
    def _parse_loop_cfg(cls, loop_cfg: dict) -> dict:
        if not isinstance(loop_cfg, dict):
            raise TypeError(f"'loop_cfg' must be dict, got {type(loop_cfg)}")

        capture_signal = loop_cfg.get("capture_signal", None)
        if capture_signal is None:
            raise ValueError("'loop_cfg.capture_signal' is required, e.g. 'reader.raw_data'")

        capture_slot = loop_cfg.get("capture_slot", "pcd_buffering_worker.enqueue_raw")
        connection_type = loop_cfg.get("capture_connection_type", "QueuedConnection")

        return dict(
            capture_signal=cls._alias_path(capture_signal),
            capture_slot=cls._alias_path(capture_slot),
            capture_connection_type=connection_type,
        )

    @property
    def capture_signal(self):
        return self._resolve_attr(self.loop_cfg["capture_signal"])

    @property
    def capture_slot(self):
        return self._resolve_attr(self.loop_cfg["capture_slot"])

    @property
    def capture_connection_type(self):
        return getattr(
            Qt.ConnectionType,
            self.loop_cfg["capture_connection_type"],
            Qt.ConnectionType.AutoConnection,
        )

    def _make_visualizer(self, vis_cfg: dict) -> "OnlineSkeletonVisualizer":
        visualizer = VISUALIZERS.build(
            dict(
                vis_cfg,
                on_close=self._on_visualizer_close,
            )
        )
        return visualizer

    def _on_visualizer_close(self, event):
        if self._controller is not None:
            self._controller.stop_all()
        else:
            self._cleanup()

    def start(self):
        self._controller = _LoopController(self)
        logger.info("Starting HPE Dataset Collection Application (v2)")
        self._setup_connections()
        self.visualizer.show()
        self._controller.start_all()
        sys.exit(self.app.exec())

    @classmethod
    def from_cfg(cls, cfg: dict):
        thread_cfg = cfg.get("thread_cfg", None)
        if thread_cfg is None:
            raise ValueError("'thread_cfg' is required for HPEDatasetCollectionAppV2.")

        if "connections" not in cfg:
            raise ValueError("'connections' is required for HPEDatasetCollectionAppV2.")

        raw_connections = cfg.get("connections", [])
        parsed_connections = cls._parse_collection_connections(raw_connections)
        parsed_loop_cfg = cls._parse_loop_cfg(cfg.get("loop_cfg", {}))

        return cls(
            thread_cfg=thread_cfg,
            vis_cfg=cfg["vis_cfg"],
            instructions=cfg.get("instructions", {}),
            buffer_cfg=cfg["buffer_cfg"],
            kinect_cfg=cfg["kinect_cfg"],
            loop_cfg=parsed_loop_cfg,
            connections=parsed_connections,
            cfg=cfg,
        )


class _LoopController(QObject):
    def __init__(self, app: HPEDatasetCollectionAppV2):
        super().__init__()
        self.app = app

        self.metainfo_popup = InputPopupDialog(
            labels=["Participant ID", "Game", "Position X", "Position Y", "Speed", "Description"]
        )

    @Slot()
    def before_init(self):
        popup = self.metainfo_popup
        popup.refresh_fields()
        popup.open()

    @Slot(int)
    def on_frame_count(self, count: int):
        self.app.visualizer.update_label(f"Frames: {count:04d}")

    @Slot()
    def on_init_stage_complete(self):
        app = self.app
        if not app.kinect_mgr_worker._running:
            app.kinect_mgr_worker.startSkeletonCaptureSignal.emit()

        timer = QTimer(self)
        timer.setInterval(2000)

        def _check():
            if app.kinect_mgr_worker._running:
                timer.stop()
                app.capture_signal.connect(app.capture_slot, app.capture_connection_type)
                app.kinect_mgr_worker.resumeSkeletonCaptureSignal.emit()
                app.instruction_worker.instructionsOnStart.emit()
            else:
                logger.info("Waiting for Kinect Manager to start…")

        timer.timeout.connect(_check)
        timer.start()

    @Slot()
    def on_buffer_full(self):
        logger.info("Buffer is full. Dumping buffered capture data.")
        self.app.pcd_buffering_worker.dump_buffer()
        self.app.capture_signal.disconnect(self.app.capture_slot)
        self.app.instruction_worker.instructionsOnStop.emit(True)

    @Slot()
    def on_cycle_complete(self):
        logger.info("Cycle complete. Preparing for next capture.")
        QMetaObject.invokeMethod(
            self.app.pcd_buffering_worker,
            "clear_buffer",
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
        ):
            t.start()

        self.before_init()

    def stop_all(self):
        for tn in (
            "reader_thread",
            "kinect_mgr_thread",
            "instruction_thread",
            "pcd_buffering_thread",
        ):
            logger.info(f"Requesting interruption for thread: app.{tn}")
            t: QThread = getattr(self.app, tn)
            t.requestInterruption()
            logger.info(f"Quitting thread: app.{tn}")
            t.quit()
            logger.info(f"Waiting for thread: app.{tn} to quit")
            t.wait()
