import threading
from typing import Optional
from apps.common.loops.onlineReader import OnlineReaderLoop, TestingLoop
from apps.common.pcd.pointCloud import SimplePointCloud5D
from apps.pc_collection.gui.main_window import DataCollectorMainWindow
from apps.pc_collection.pc_buffer import PointCloudBuffer
import kinect_toolkits as kntk

from radario.base import BaseBufferedReader


class OnlineDataCollectionLoop(OnlineReaderLoop):
    def __init__(self,
                 reader: BaseBufferedReader,
                 interval: float,
                 buffer: PointCloudBuffer,
                 gui: DataCollectorMainWindow,
                 kinect_cfg: Optional[dict] = None):
        super().__init__(reader, interval)
        self.buffer = buffer
        self.gui = gui
        self.kinect_manager = None if kinect_cfg is None else kntk.KinectManager.from_dict(kinect_cfg)
        
        def _on_frame_arrival(_buffer: PointCloudBuffer):
            self.gui._on_frame_arrival(_buffer)
            if self.kinect_manager is None:
                return
            last_skeleton = self.kinect_manager.get_last_record()
            self.gui.refresh_skel_visual(last_skeleton)
            
        self.buffer.set_on_frame_arrival(_on_frame_arrival)
        self.buffer.set_on_buffer_full(self._on_buffer_full)
        self.first_iter = True

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
            gui=cfg.get("gui"),
            kinect_cfg=cfg.get("kinect_cfg", None)
        )
    
    def start(self):
        if self.first_iter:
            self.first_iter = False
            self.gui.show_break_popup(self.start)
        else:
            if self.kinect_manager:
                print("Starting kinect")
                self.kinect_manager.start_skeleton_capture()
                self.kinect_manager.wait_for_capture_starts()
            super().start()
            
    def stop(self):
        super().stop()
        if self.kinect_manager:
            self.kinect_manager.stop_skeleton_capture(wait=1)
            if self.buffer.recent_dump is not None:
                self.kinect_manager.rename_default_output_file(self.buffer.recent_dump)
                self.buffer.recent_dump = None
            else:
                self.kinect_manager.delete_default_output_file()
                pass
        
    def _on_buffer_full(self, buffer: PointCloudBuffer):
        self.stop()
        print("Buffer full")
        buffer.dump()
        self.gui.show_break_popup(self.start)
        
    def _after_stop_hook(self):
        super()._after_stop_hook()