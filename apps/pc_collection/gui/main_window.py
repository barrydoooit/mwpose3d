from functools import partial
import tkinter as tk
from typing import Any, Callable, Optional, Tuple

from apps.common.pcd.pointCloudVis import PointCloudFigureFrame
from apps.pc_collection.gui.cli import CommandProcessor
from apps.pc_collection.gui.popups.calib_instruction import TimeCalibInstructionPopup
from apps.pc_collection.gui.popups.midbreak import MidBreakPopup
from apps.common.pcd.pc_buffer import PointCloudBuffer
from kinect_toolkits.kinectData import Skeleton
from kinect_toolkits.kinectVis import SkeletonFigureFrame

class DataCollectorMainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.break_popup = None
        self.time_calib_popup = None
        
        self._setup_ui()
    
    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(tk.Tk())
    
    def bind_buffer(self, pcd_buffer: PointCloudBuffer):
        self.command_processor.pcd_buffer = pcd_buffer
        
    def _setup_ui(self):
        self.root.title("Data Collector")
        
        self.frame_counter_label = tk.Label(self.root, text="Current Frame: 0", font=("Arial", 24))
        self.frame_counter_label.pack(side=tk.TOP, fill=tk.X)
        
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.cli_frame = tk.Frame(main_frame, width=200)
        self.cli_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.cli_text = tk.Text(self.cli_frame, height=15, width=30)
        self.cli_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.cmd_entry = tk.Entry(self.cli_frame)
        self.cmd_entry.pack(side=tk.BOTTOM, fill=tk.X)
        
        vis_container = tk.Frame(main_frame)
        vis_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.pcd_vis_frame = PointCloudFigureFrame(
            parent=vis_container,
            figure_cfg=dict(
                figsize=(5, 5),
                dpi=100
            )
        )
        self.pcd_vis_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.skel_vis_frame = SkeletonFigureFrame(
            master=vis_container,
            figure_cfg=dict(
                figsize=(5, 5),
                dpi=100
            )
        )
        self.skel_vis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
    def bind_cli_processor(self, cli_processor: CommandProcessor):
        self.cmd_entry.bind("<Return>", partial(self._on_command_entered, cli_processor))
        
    def _on_command_entered(self, cmd_processor: CommandProcessor, event):
        cmd = self.cmd_entry.get()
        self.cmd_entry.delete(0, tk.END)
        result = cmd_processor.process(cmd)
        self.cli_text.insert(tk.END, f"> {cmd}\n{result}\n")
        self.cli_text.see(tk.END)
    
    def _on_frame_arrival(self, pcd_buffer: PointCloudBuffer):
        self.root.after(0, lambda: self.refresh_pcd_visual(pcd_buffer))
    
    def refresh_pcd_visual(self, pcd_buffer: PointCloudBuffer):
        self.frame_counter_label.config(text=f"Current Frame: {pcd_buffer._frame_counter}")
        with pcd_buffer._buffer_lock:
            if len(pcd_buffer.buffer) == 0:
                return
            frame = pcd_buffer.buffer[-1]
        self.pcd_vis_frame.update(frame.point_cloud.points)
    
    def refresh_skel_visual(self, skeleton: Skeleton):
        if skeleton is None:
            return
        self.skel_vis_frame.update(skeleton)
        
    def reset_all_visuals(self):
        self.reset_pcd_visual()
        self.reset_skel_visual()
    
    def reset_pcd_visual(self):
        self.frame_counter_label.config(text="Current Frame: 0")
        self.pcd_vis_frame.reset()

    def reset_skel_visual(self):
        self.skel_vis_frame.reset()
        
    def show_break_popup(self, break_time: int = 10,
                         on_break_end: Optional[callable] = None):
        if self.break_popup is not None:
            return
        self.remaining_break_time = break_time
        self.break_popup = MidBreakPopup(self.root, 
                                         break_time, 
                                         on_break_end=on_break_end)
        self.break_popup.transient(self.root)  # Set the popup to be transient to the main window
        self.break_popup.lift()               # Bring the popup to the front
        self.break_popup.grab_set()           # Optional: ensure all events are directed to the popup
        self.break_popup.protocol("WM_DELETE_WINDOW", self.break_popup.close_break_popup)

    def show_time_calib_popup(self, 
                              pcd_buffer: "PointCloudBuffer",
                              stages_duration: Tuple[int, int, int, int],
                              ticks_to_confirm: int,
                              before_popup_close: Optional[Callable[["TimeCalibInstructionPopup"], Any]] = None):
        if self.time_calib_popup is not None:
            return
        self.time_calib_popup = TimeCalibInstructionPopup(self.root, 
                                                     pcd_buffer, 
                                                     stages_duration, 
                                                     ticks_to_confirm,
                                                     before_popup_close=before_popup_close)
        self.time_calib_popup.protocol("WM_DELETE_WINDOW", self.time_calib_popup.close_popup)
    
    def stop_all_popups(self):
        if self.break_popup:
            self.break_popup.close_break_popup(with_callback=False)
            self.break_popup = None
        if self.time_calib_popup:
            self.time_calib_popup.close_popup(with_callback=False)
            self.time_calib_popup = None
    
class DataCollectorDelegate:
    pass