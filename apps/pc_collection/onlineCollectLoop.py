import threading
import time
from typing import Optional
from apps.common.loops.onlineReader import OnlineReaderLoop, TestingLoop
from apps.common.pcd.pointCloud import SimplePointCloud5D
from apps.pc_collection.gui.main_window import DataCollectorMainWindow
from apps.pc_collection.gui.popups.calib_instruction import TimeCalibInstructionPopup
from apps.pc_collection.pc_buffer import PointCloudBuffer
import kinect_toolkits as kntk

from radario.base import BaseBufferedReader


class OnlineDataCollectionLoop(OnlineReaderLoop):
    def __init__(self,
                 reader: BaseBufferedReader,
                 interval: float,
                 break_time: int,
                 buffer: PointCloudBuffer,
                 gui: DataCollectorMainWindow,
                 time_calib: bool ,
                 kinect_cfg: Optional[dict] = None):
        super().__init__(reader, interval)
        self.buffer = buffer
        self.gui = gui
        self.kinect_manager = None if kinect_cfg is None else kntk.KinectManager.from_dict(kinect_cfg)
        self.break_time = break_time
        def _on_frame_arrival(_buffer: PointCloudBuffer):
            self.gui._on_frame_arrival(_buffer)
            if self.kinect_manager is None:
                return
            def update_skeleton():
                last_skeleton = self.kinect_manager.get_last_record()
                self.gui.refresh_skel_visual(last_skeleton)
            threading.Thread(target=update_skeleton, daemon=True).start()
            
        self.buffer.set_on_frame_arrival(_on_frame_arrival)
        self.buffer.set_on_buffer_full(self._on_buffer_full)
        
        self.first_iter = True
        self.time_calib = time_calib

    def _process_data(self, data):
        data_ok, frame_number, det_obj = data
        if not data_ok:
            return
        # print(f"Num Objects Detected: {det_obj['numObj']}")
        self.buffer.add_frame(SimplePointCloud5D.from_dict(det_obj))

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(
            reader=cfg.get("reader"),
            interval=cfg.get("interval"),
            break_time=cfg.get("break_time"),
            buffer=cfg.get("buffer"),
            gui=cfg.get("gui"),
            time_calib=cfg.get("time_calib", True),
            kinect_cfg=cfg.get("kinect_cfg", None)
        )
    
    def start(self):
        if self.first_iter:
            self.first_iter = False
            if self.runner._mode.value == "collect":
                self.gui.show_break_popup(self.break_time, self.start)
        else:                
            if self.kinect_manager:
                print("Starting kinect")
                self.kinect_manager.start_skeleton_capture()
                self.kinect_manager.wait_for_capture_starts()
            super().start()
            if self.time_calib:
                self.start_ts = time.time()
                self.gui.show_time_calib_popup(
                    pcd_buffer=self.buffer,
                    stages_duration=(4, 20, 2, 1,),
                    ticks_to_confirm=4,
                    before_popup_close=self._on_time_calib_end
                )
            
    def stop(self):
        super().stop()
        self.gui.stop_all_popups()
        if self.kinect_manager:
            self.kinect_manager.stop_skeleton_capture(wait=5)
            if self.runner._mode.value == "visualize":
                self.kinect_manager.delete_default_output_file()
        
    def _after_stop_hook(self):
        super()._after_stop_hook()
    
            
    def _on_buffer_full(self, buffer: PointCloudBuffer):
        self.stop()
        print("Buffer full")
        buffer.dump()
        if self.buffer.recent_dump is not None:
            self.kinect_manager.rename_default_output_file(self.buffer.recent_dump)
            self.buffer.recent_dump = None
        self.gui.show_break_popup(self.break_time, self.start)
    
    def _on_time_calib_end(self, time_calib_popup: "TimeCalibInstructionPopup"):
        still_ts = time_calib_popup.still_period
        pcd_buffer_now_size = self.buffer.enlarge_buffer(len(self.buffer) + self.buffer.max_buffer_size)
        self.buffer.add_metadata("still_start_ts", still_ts[0])\
            .add_metadata("still_end_ts", still_ts[1])\
            .add_metadata("calib_frames", pcd_buffer_now_size)\
            .add_metadata("radar_start_ts", self.start_ts)