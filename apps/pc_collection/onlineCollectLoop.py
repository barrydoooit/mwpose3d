import threading
from apps.common.loops.onlineReader import OnlineReaderLoop, TestingLoop
from apps.common.pcd.pointCloud import SimplePointCloud5D
from apps.pc_collection.gui.main_window import DataCollectorMainWindow
from apps.pc_collection.pc_buffer import PointCloudBuffer
from radario.base import BaseBufferedReader


class OnlineDataCollectionLoop(OnlineReaderLoop):
    def __init__(self,
                 reader: BaseBufferedReader,
                 interval: float,
                 buffer: PointCloudBuffer,
                 gui: DataCollectorMainWindow):
        super().__init__(reader, interval)
        self.buffer = buffer
        self.gui = gui
        self.buffer.set_on_frame_arrival(self.gui._on_frame_arrival)
        self.buffer.set_on_buffer_full(self._on_buffer_full)

    def _process_data(self, data):
        data_ok, frame_number, det_obj = data
        if not data_ok:
            return
        self.buffer.add_frame(SimplePointCloud5D.from_dict(det_obj))

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(
            reader=cfg.get("reader"),
            interval=cfg.get("interval"),
            buffer=cfg.get("buffer"),
            gui=cfg.get("gui")
        )
    
    def _on_buffer_full(self, buffer: PointCloudBuffer):
        self.stop()
        print("Buffer full")
        buffer.dump()
        self.gui.show_break_popup(self.start)
        
    def _after_stop_hook(self):
        super()._after_stop_hook()