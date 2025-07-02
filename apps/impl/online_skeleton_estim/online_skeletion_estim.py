from pathlib import Path
import sys
from typing import Optional, Union, TYPE_CHECKING
from apps.impl.online_skeleton_estim.estimation_worker import InferenceWorkerThread
from mwcore.apps import BaseMWOnlineApp
from mwcore.registry import APPS
from mwpose3d.registry import VISUALIZERS
from mwcore.visualization import OnlinePointCloudVisualizer
from mwpose3d.runner.inference_engine import InferenceEngine
from mmengine.config import Config
from PySide6.QtCore import Qt, QCoreApplication
QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@APPS.register_module()
class OnlineSkeletionEstimationApp(BaseMWOnlineApp):
    def __init__(self,
                 reader_cfg: dict,
                 hpe_model_cfg: Union[Path, str, dict],
                 vis_cfg: dict,
                 load_from: Optional[Union[Path, str]] = None):
        super().__init__(reader_cfg, vis_cfg)
        self.hpe_model_cfg = self._load_hpe_cfg(hpe_model_cfg, load_from)
    
    @staticmethod
    def _load_hpe_cfg(hpe_model_cfg: Union[Path, str, dict], load_from: Optional[str]) -> dict:
            cfg = hpe_model_cfg if isinstance(hpe_model_cfg, dict) else Config.fromfile(hpe_model_cfg)
            if load_from is not None:
                cfg.load_from = load_from
                logger.info(f"Loading checkpoint from {load_from}")
            return cfg

    @property
    def inference_engine(self) -> InferenceEngine:
        if not hasattr(self, '_inference_engine'):
            self._inference_engine = InferenceEngine.from_cfg(self.hpe_model_cfg)
            logger.info(f"Loaded inference engine with model: {self._inference_engine.model.__class__.__name__}")
        return self._inference_engine
    
    @property
    def inference_thread(self) -> InferenceWorkerThread:
        if not hasattr(self, '_inference_thread'):
            self._inference_thread = InferenceWorkerThread(self, parent=None)
            if hasattr(self.visualizer, 'update_skeleton'):
                self._inference_thread.inferece_done.connect(self.visualizer.update_skeleton)
            logger.info("Inference worker thread initialized.")
        return self._inference_thread

    def _make_visualizer(self, vis_cfg: dict) -> 'OnlinePointCloudVisualizer':
        def _on_close(event):
            self.reader_thread.requestInterruption()
            self.reader_thread.wait()
        visualizer = VISUALIZERS.build(dict(
            vis_cfg,
            on_close=_on_close
        ))
        assert isinstance(visualizer, OnlinePointCloudVisualizer)
        return visualizer

    def start(self):
        self.app = QApplication(sys.argv)
        logger.info("Starting Online Skeleton Estimation Application...")
        self.reader_thread.array_data.connect(self.inference_thread.enqueue, Qt.ConnectionType.QueuedConnection)
        self.reader_thread.array_data.connect(self.visualizer.on_new_cloud, Qt.ConnectionType.QueuedConnection)
        self.inference_thread.start()
        self.reader_thread.start()
        self.visualizer.show()
        sys.exit(self.app.exec())

    @classmethod
    def from_cfg(cls, cfg: dict):
        reader_cfg = cfg.get('reader_cfg')
        hpe_model_cfg = cfg.get('hpe_model_cfg')
        vis_cfg = cfg.get('vis_cfg')
        load_from = cfg.get('load_from')
        return cls(reader_cfg, hpe_model_cfg, vis_cfg, load_from)
