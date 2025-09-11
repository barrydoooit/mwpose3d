from collections import deque
from functools import partial
import mmap
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition, Slot, QObject, QMutexLocker
import numpy as np
from typing import TYPE_CHECKING, Deque, List, Optional

from mwpose3d.utils.kinect_toolkits.kinectData import KeypointType
if TYPE_CHECKING:
    from .online_skeletion_estim import OnlineSkeletionEstimationApp
    from mwpose3d.runner.inference_engine import InferenceEngine



class InferenceWorker(QObject):
    inference_done = Signal(np.ndarray)
    error = Signal(str)
    busy_changed = Signal(bool)

    _default_ctrl_joints = [
         KeypointType.SPINE_BASE,
         KeypointType.ELBOW_RIGHT,
         KeypointType.WRIST_RIGHT,
         KeypointType.ELBOW_LEFT,
         KeypointType.WRIST_LEFT,
     ]
    
    @classmethod
    def build_with_thread(cls, app, config, parent=None):
        thread = QThread(parent=parent)
        worker = cls(
            app=app,
            with_ctrl_joints=config.get('with_ctrl_joints', True),
            ctrl_joints=config.get('ctrl_joints', None),
            max_concurrency=config.get('max_concurrency', 2),
            queue_capacity=config.get('queue_capacity', 1),
        )
        worker.moveToThread(thread)
        thread.finished.connect(worker.deleteLater)
        return worker, thread

    def __init__(
            self,
            app: "OnlineSkeletionEstimationApp",
            *,
            with_ctrl_joints: bool = True,
            ctrl_joints: Optional[List[int | KeypointType]] = None,
            max_concurrency: int = 1,
            queue_capacity: int = 1,
            serialize_engine: bool = False,
            parent: Optional[QObject] = None,
            ) -> None:
        super().__init__(parent)
        self.app = app
        self.with_ctrl_joints = with_ctrl_joints


        # Concurrency controls
        self._mutex = QMutex()
        self._inflight = 0
        self._max_concurrency = max(1, int(max_concurrency))
        self._queue_capacity = max(self._max_concurrency, int(queue_capacity))
        self._queue: Deque[object] = deque(maxlen=self._queue_capacity)
        self._pool = ThreadPoolExecutor(max_workers=self._max_concurrency, thread_name_prefix="infer")
        self._serialize_engine = bool(serialize_engine)
        self._engine_lock = threading.Lock()


        # Engine / KP mapping state
        self._packet_dtype = np.float32
        self._kp_index_map: Optional[np.ndarray] = None


        # Controller shared memory state
        self._controller_lock = threading.Lock()
        self._ctrl_joints: List[KeypointType] = []
        self._controller_packet_size = 0
        self._controller_mmf: Optional[mmap.mmap] = None
        self.update_ctrl_joints(ctrl_joints)
        self._init_controller_mmf()

        # ADDED: Update-flag MMF (1 byte) to signal new frames have been written.
        self._update_mmf: Optional[mmap.mmap] = None
        self._update_lock = threading.Lock()
        try:
            _update_map_name = r"Local\KinectUpdateFlag"
            # create a 1-byte MMF for signalling; if creation fails, leave as None
            self._update_mmf = mmap.mmap(-1, 1, tagname=_update_map_name, access=mmap.ACCESS_WRITE)
            # initialize to zero
            try:
                with self._update_lock:
                    self._update_mmf.seek(0)
                    self._update_mmf.write(b'\x00')
            except Exception:
                # ignore initialization errors
                pass
        except Exception as e:
            # if creating the update mmf fails, keep it None and emit an error optionally
            try:
                self.error.emit(f"Could not create update MMF: {e}")
            except Exception:
                pass
        # Ordering state
        self._seq_next = 0
        self._emit_next = 0
        self._results: dict[int, Optional[np.ndarray]] = {}
        self._order_lock = threading.Lock()

        # Keep a small ring of recent frames to build per-task windows (no copies)
        self._queue: Deque[object] = deque()

        self._window_size: Optional[int] = None
        self._frame_ring: Deque[object] = deque(maxlen=1)  # temporary; real size set lazily

        self.last_input_frame_time = 0.0
        self.last_output_frame_time = 0.0

        
    # ---------- Public API ----------
    @property
    def inference_engine(self) -> "InferenceEngine":
        return self.app.inference_engine

    def update_ctrl_joints(self, ctrl_joints: Optional[List[int | KeypointType]]):
        if not self.with_ctrl_joints:
            self._ctrl_joints = []
        else:
            if not ctrl_joints:
                self._ctrl_joints = self._default_ctrl_joints
            else:
                self._ctrl_joints = [KeypointType(int(kp.value if hasattr(kp, "value") else kp)) for kp in ctrl_joints]
        # 3 floats per joint
        self._controller_packet_size = 3 * 4 * len(self._ctrl_joints)
        self._kp_index_map = None  # force rebuild against engine's kp order

    def _init_controller_mmf(self) -> None:
        try:
            if self._controller_mmf is not None:
                self._controller_mmf.close()
        except Exception:
            pass
        self._controller_mmf = None
        if self._controller_packet_size > 0:
            _controller_map_name = r"Local\KinectControl"
            self._controller_mmf = mmap.mmap(
                -1,
                self._controller_packet_size,
                tagname=_controller_map_name,
                access=mmap.ACCESS_WRITE,
            )
    def _ensure_window_init(self):
        if self._window_size is not None:
            return
        # Access the engine here so it’s created in the worker thread
        try:
            self._window_size = int(getattr(self.app.inference_engine, "frame_buffer_size", 1))
        except Exception:
            self._window_size = 1
        # rebuild ring to the right size
        self._frame_ring = deque(maxlen=self._window_size)

    @Slot(object)
    def enqueue(self, frame: object) -> None:
        self._ensure_window_init()
        if isinstance(frame, np.ndarray):
            frame = frame.copy()
        # print("Interval since last input frame: {:.3f} s".format(
        #     time.perf_counter() - self.last_input_frame_time
        # ))
        self.last_input_frame_time = time.perf_counter()
        # 1) Extend ring with newest frame
        self._frame_ring.append(frame)

        # 2) Warm-up: don't enqueue until the window is full
        if len(self._frame_ring) < self._window_size:
            return

        # 3) Snapshot the window (cheap tuple of refs)
        window_snapshot = tuple(self._frame_ring)

        # 4) Push into bounded queue with tail-replacement coalescing
        with QMutexLocker(self._mutex):
            if len(self._queue) >= self._queue_capacity:
                # Replace the newest pending task (maintains order, avoids left-drop)
                self._queue[-1] = window_snapshot
            else:
                self._queue.append(window_snapshot)

            # Collect work; submission happens outside the lock in _maybe_dispatch_locked
            self._maybe_dispatch_locked()


    @Slot()
    def stop(self) -> None:
        """Close resources and stop the pool. Call before quitting the thread."""
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            self.error.emit(f"Error shutting down pool: {e}")
        try:
            with self._controller_lock:
                if self._controller_mmf is not None:
                    self._controller_mmf.close()
                    self._controller_mmf = None
        except Exception as e:
            self.error.emit(f"Error closing controller MMF: {e}")
        # ADDED: close update mmf
        try:
            with self._update_lock:
                if self._update_mmf is not None:
                    self._update_mmf.close()
                    self._update_mmf = None
        except Exception as e:
            self.error.emit(f"Error closing update MMF: {e}")

    # ---------- Internals ----------
    def _ensure_kp_index_map(self) -> None:
        if self._kp_index_map is not None:
            return
        eng = self.inference_engine
        kp_order = list(eng.keypoints_involved)  # sequence of ints
        idxs: list[int] = []
        for kp in self._ctrl_joints:
            val = kp.value if hasattr(kp, "value") else int(kp)
            try:
                idxs.append(kp_order.index(val))
            except ValueError as e:
                raise RuntimeError(f"Keypoint {val} not in engine keypoints_involved") from e
        self._kp_index_map = np.asarray(idxs, dtype=np.int32)

    def _write_controller_mmf(self, result: np.ndarray) -> None:
        if not self.with_ctrl_joints or len(self._ctrl_joints) == 0:
            return
        if self._controller_mmf is None:
            return
        self._ensure_kp_index_map()
        arr = np.asarray(result)
        coords = arr.reshape(-1, 3)
        sel = coords[self._kp_index_map]

        # Transform axes (match original impl)
        packet = np.empty_like(sel, dtype=self._packet_dtype)
        packet[:, 0] = -sel[:, 0]
        packet[:, 1] = sel[:, 2]
        packet[:, 2] = sel[:, 1]

        with self._controller_lock:
            self._controller_mmf.seek(0)
            self._controller_mmf.write(memoryview(packet.astype(self._packet_dtype, copy=False)).cast("B"))

        # ADDED: signal update MMF (write single byte = 1) after controller MMF write
        if self._update_mmf is not None:
            try:
                with self._update_lock:
                    self._update_mmf.seek(0)
                    self._update_mmf.write(b'\x01')
            except Exception as e:
                # don't let update-flag failures break flow; emit error
                try:
                    self.error.emit(f"Error writing update MMF: {e}")
                except Exception:
                    pass

    def _run_inference(self, window_snapshot: tuple) -> Optional[np.ndarray]:
        try:
            if self._serialize_engine:
                with self._engine_lock:
                    out = self.inference_engine.infer_window(window_snapshot)
            else:
                out = self.inference_engine.infer_window(window_snapshot)
            # print("Interval since last output frame: {:.3f} s".format(
            #     time.perf_counter() - self.last_output_frame_time
            # ))
            self.last_output_frame_time = time.perf_counter()
        except Exception as e:
            self.error.emit(str(e))
            out = None
        return out

    def _maybe_dispatch_locked(self) -> None:
        # assumes self._mutex is locked
        to_submit = []
        was_idle = self._inflight == 0

        slots = self._max_concurrency - self._inflight
        n = min(slots, len(self._queue))
        for _ in range(n):
            to_submit.append(self._queue.popleft())
        self._inflight += n

        if was_idle and n > 0:
            self.busy_changed.emit(True)

        # Submit outside the lock
        for payload in to_submit:
            fut = self._pool.submit(self._run_inference, payload)
            # if you're using the in-order emission I suggested earlier, wire the seq there
            fut.add_done_callback(self._on_task_done)

    # Runs in a pool thread
    def _on_task_done(self, fut: Future) -> None:
        try:
            result = fut.result()
        except Exception as e:
            result = None
            self.error.emit(f"Inference task error: {e}")

        if result is not None:
            try:
                self._write_controller_mmf(result)
            except Exception as e:
                self.error.emit(f"Controller write failed: {e}")
            self.inference_done.emit(result)

        # minimal critical section
        with QMutexLocker(self._mutex):
            self._inflight -= 1
            became_idle = self._inflight == 0
            # try to collect more work (doesn't submit here)
            self._maybe_dispatch_locked()
        if became_idle:
            self.busy_changed.emit(False)

    def _drain_ready_locked(self) -> None:
        # Assumes self._order_lock is held
        while self._emit_next in self._results:
            result = self._results.pop(self._emit_next)
            if result is not None:
                try:
                    self._write_controller_mmf(result)
                except Exception as e:
                    self.error.emit(f"Controller write failed: {e}")
                self.inference_done.emit(result)
            # Even if result is None (e.g. not enough frames yet), advance the sequence
            self._emit_next += 1

class InferenceWorkerThread(QThread):
    inference_done = Signal(np.ndarray)

    _default_ctrl_joints = [
         KeypointType.SPINE_MID,
         KeypointType.ELBOW_RIGHT,
         KeypointType.WRIST_RIGHT,
         KeypointType.ELBOW_LEFT,
         KeypointType.WRIST_LEFT,
     ]
    
    def __init__(self,  
                 app: 'OnlineSkeletionEstimationApp',
                 with_ctrl_joints: bool = True,
                 ctrl_joints: list[int] = None,
                 parent=None):
        super().__init__(parent)
        self.app = app
        self.with_ctrl_joints = with_ctrl_joints

        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._latest_frame = None
        self._running = False

        self._packet_dtype = np.float32
        self._kp_index_map: np.ndarray = None
        self._controller_lock = threading.Lock()

        self._ctrl_joints: List[KeypointType] = []
        self.update_ctrl_joints(ctrl_joints)
        
        _controller_map_name = r"Local\KinectControl"
        self._controller_mmf = mmap.mmap(
            -1, self._controller_packet_size, tagname=_controller_map_name, access=mmap.ACCESS_WRITE
        )
        self._controller_lock = threading.Lock()

        # ADDED: create update-flag MMF (1 byte) to signal new frames
        self._update_lock = threading.Lock()
        self._update_mmf: Optional[mmap.mmap] = None
        try:
            _update_map_name = r"Local\KinectUpdateFlag"
            self._update_mmf = mmap.mmap(-1, 1, tagname=_update_map_name, access=mmap.ACCESS_WRITE)
            try:
                with self._update_lock:
                    self._update_mmf.seek(0)
                    self._update_mmf.write(b'\x00')
            except Exception:
                pass
        except Exception as e:
            # best-effort: print since this thread class doesn't have an error signal
            try:
                print(f"Could not create update MMF: {e}")
            except Exception:
                pass
        
        self.last_input_frame_time = 0.0
        self.last_output_frame_time = 0.0
        
    def update_ctrl_joints(self, ctrl_joints: list[KeypointType]):
        if not self.with_ctrl_joints:
            self._ctrl_joints = []
        else:
            if not ctrl_joints or len(ctrl_joints) == 0:
                self._ctrl_joints = self._default_ctrl_joints
            else:
                self._ctrl_joints = [KeypointType(kp) for kp in ctrl_joints]
        
        self._controller_packet_size = 3 * 4 * len(self._ctrl_joints)
        self._kp_index_map = None
    
    @property
    def inference_engine(self) -> 'InferenceEngine':
        return self.app.inference_engine

    def _ensure_kp_index_map(self):
        if self._kp_index_map is not None:
            return
        eng = self.inference_engine
        kp_order = list(eng.keypoints_involved)
        idxs = []
        for kp in self._ctrl_joints:
            val = kp.value if hasattr(kp, 'value') else int(kp)
            idxs.append(kp_order.index(val))
        self._kp_index_map = np.array(idxs, dtype=np.int32)
            
    @Slot(object)
    def enqueue(self, frame):
        print("Interval since last input frame: {:.3f} s".format(
            time.perf_counter() - self.last_input_frame_time
        ))
        self.last_input_frame_time = time.perf_counter()
        self._mutex.lock()
        self._latest_frame = frame
        self._cond.wakeOne()
        self._mutex.unlock()
    
    def write_controller_mmf(self, result: np.ndarray):
        if not self.with_ctrl_joints or len(self._ctrl_joints) == 0:
            return
        self._ensure_kp_index_map()

        arr = np.asarray(result)
        coords = arr.reshape(-1, 3)
        sel = coords[self._kp_index_map]

        packet = np.empty_like(sel, dtype=self._packet_dtype)
        packet[:, 0] = -sel[:, 0]
        packet[:, 1] = sel[:, 2]
        packet[:, 2] = sel[:, 1]

        with self._controller_lock:
            self._controller_mmf.seek(0)
            self._controller_mmf.write(
                memoryview(packet.astype(self._packet_dtype, copy=False)).cast("B")
            )

        # ADDED: signal update MMF (write single byte = 1) after controller MMF write
        if getattr(self, "_update_mmf", None) is not None:
            try:
                with self._update_lock:
                    self._update_mmf.seek(0)
                    self._update_mmf.write(b'\x01')
            except Exception as e:
                # best-effort logging
                try:
                    print(f"Error writing update MMF: {e}")
                except Exception:
                    pass

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
            print("Interval since last output frame: {:.3f} s".format(
                time.perf_counter() - self.last_output_frame_time
            ))
            self.last_output_frame_time = time.perf_counter()
            if result is not None:
                self.write_controller_mmf(result)
                self.inference_done.emit(result)

        QThread.currentThread().quit()
    
    def stop(self):
        self._mutex.lock()
        self._running = False
        self._cond.wakeOne()
        self._mutex.unlock()
        self.wait()
