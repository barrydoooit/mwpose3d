from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition, Slot
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .online_skeletion_estim import OnlineSkeletionEstimationApp
    from mwpose3d.runner.inference_engine import InferenceEngine



class InferenceWorkerThread(QThread):
    inferece_done = Signal(np.ndarray)

    def __init__(self, 
                 app: 'OnlineSkeletionEstimationApp',
                 parent=None):
        super().__init__(parent)
        self.app = app
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._latest_frame = None
        self._running = False
    
    @property
    def inference_engine(self) -> 'InferenceEngine':
        return self.app.inference_engine

    @Slot(object)
    def enqueue(self, frame):
        self._mutex.lock()
        self._latest_frame = frame
        self._cond.wakeOne()
        self._mutex.unlock()
    
    def run(self):
        self._running = True
        while True:
            self._mutex.lock()

            while self._running and self._latest_frame is None:
                self._cond.wait(self._mutex)
            
            if not self._running:
                self._mutex.unlock()
                break

            frame = self._latest_frame
            self._latest_frame = None
            self._mutex.unlock()

            result = self.inference_engine.infer(frame)
            if result is not None:
                self.inferece_done.emit(result)
        
        QThread.currentThread().quit()
    
    def stop(self):
        self._mutex.lock()
        self._running = False
        self._cond.wakeOne()
        self._mutex.unlock()
        self.wait()
