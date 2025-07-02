import time
from typing import List, Dict, Any, Callable, Tuple
import logging

from PySide6.QtCore import (
    QObject,
    QThread,
    Signal,
    Slot,
    QTimer,
    QMetaObject, Qt
)
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout

logger = logging.getLogger(__name__)



class InstructionPopup(QDialog):
    nextMessageRequested = Signal(str) # Stage

    def __init__(self, **kwargs):
        super().__init__()
        self.setWindowTitle("Instruction")
        self.label = QLabel("")
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)

        if 'style_sheet' in kwargs:
            self.setStyleSheet(kwargs['style_sheet'])

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    @Slot(str)
    def update_message(self, message: str):
        self.label.setText(message)
        if not self.isVisible():
            self.show()

    @Slot(str, int)
    def schedule_update_message(self, stage: str, delay_ms: int):
        if self._timer.isActive():
            self._timer.stop()
        try:
            self._timer.timeout.disconnect()
        except (TypeError, RuntimeError):
            pass
        self._timer.timeout.connect(lambda: self.nextMessageRequested.emit(stage))
        self._timer.start(delay_ms)

class InstructionWorker(QObject):
    newMessage = Signal(str)
    finishedOnInit = Signal()
    finishedOnStart = Signal()
    finishedOnStop = Signal()

    schduleUpdateMessage = Signal(str, int)
    closePopup = Signal()
    instructionsOnInit = Signal()
    instructionsOnStart = Signal()
    instructionsOnStop = Signal(bool)

    def __init__(self,
                 on_init: List[Dict[str, Any]],
                 on_start: List[Dict[str, Any]],
                 on_stop: List[Dict[str, Any]]):
        super().__init__()
        self.stages = dict(init=on_init, start=on_start, stop=on_stop)
        self.tasks = {k: self._compile_stages(v) for k, v in self.stages.items()}

        self.instructionsOnInit.connect(self._run_init)
        self.instructionsOnStart.connect(self._run_start)
        self.instructionsOnStop.connect(self._run_stop)

    def _compile_stages(self, steps: List[Dict]) -> List[Tuple[str, int]]:
        seq: List[Tuple[str, int]] = []
        for step in steps:
            contents = step.get('content','')
            durations = step.get('duration', 0)
            repeats   = step.get('repeats', 1)

            if isinstance(contents, str):
                contents = [contents]
            if isinstance(durations, (int, float)):
                durations = [durations]

            for _ in range(repeats):
                for msg, secs in zip(contents, durations):
                    seq.append((msg, int(secs * 1000)))

        return seq
    
    @property   
    def popup(self) -> InstructionPopup:
        if not hasattr(self, '_popup'):
            self._prepare_popup()
        return self._popup
    
    def _prepare_popup(self):
        self._popup = InstructionPopup(style_sheet="font-size: 32px;")
        self._popup.nextMessageRequested.connect(self._advance_stage)
        self.newMessage.connect(self._popup.update_message)
        self.schduleUpdateMessage.connect(self._popup.schedule_update_message)
        self.closePopup.connect(self._popup.close)
    
    @classmethod
    def build_with_thread(cls,
               on_init: List[Dict[str, Any]] = dict(),
               on_start: List[Dict[str, Any]] = dict(),
               on_stop: List[Dict[str, Any]] = dict()) -> 'Tuple[InstructionWorker, QThread]':
        for stage in [on_init, on_start, on_stop]:
            for step in stage:
                if not isinstance(step, dict):
                    raise ValueError(f"Instruction type incorrect: {type(step)}. Expected dict.")
                if 'content' not in step:
                    step['contect'] = ''
                if 'duration' not in step:
                    raise ValueError("Duration missing")
                if 'repeats' not in step:
                    step['repeats'] = 1
                if isinstance(step['content'], list) and isinstance(step['duration'], list):
                    if len(step['content']) != len(step['duration']):
                        raise ValueError("Length mismatch between 'content' and 'duration'.")
        
        worker = cls(on_init=on_init, on_start=on_start, on_stop=on_stop)
        thread = QThread()
        thread.finished.connect(worker.popup.close)
        thread.finished.connect(worker.popup.deleteLater)
        worker.moveToThread(thread)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        return worker, thread

    def _run_stage(self, stage: str):
        logger.info(f"Running stage: {stage}")
        self._current_seq = self.tasks[stage]  # list of (msg,delay_ms)
        self._seq_index = 0
        self._advance_stage(stage)
    
    def _advance_stage(self, stage: str):
        if QThread.currentThread().isInterruptionRequested():
            self.closePopup.emit()
            QThread.currentThread().quit()
            return

        if self._seq_index >= len(self._current_seq):
            getattr(self, f'finishedOn{stage.capitalize()}').emit()
            return

        msg, delay = self._current_seq[self._seq_index]
        self.newMessage.emit(msg)
        self._seq_index += 1

        self.schduleUpdateMessage.emit(stage, delay)

    @Slot()
    def _run_init(self):
        _ = self.popup  # Ensure popup is created
        self._run_stage('init')

    @Slot()
    def _run_start(self):
        self._run_stage('start')
    
    @Slot(bool)
    def _run_stop(self, keep_popup: bool=False):
        self._run_stage('stop')
        if not keep_popup:
            self.closePopup.emit()

