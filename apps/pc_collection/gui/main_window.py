from functools import partial
import tkinter as tk
from typing import Optional

from apps.common.pcd.pointCloudVis import PointCloudFigureFrame
from apps.pc_collection.gui.cli import CommandProcessor
from apps.pc_collection.pc_buffer import PointCloudBuffer
from kinect_toolkits.kinectData import Skeleton
from kinect_toolkits.kinectVis import SkeletonFigureFrame

class DataCollectorMainWindow:
    def __init__(self, root: tk.Tk, break_time: int = 15):
        self.root = root
        self.popup = None
        self.popup_timer = None
        self.break_time = break_time
        self.remaining_break_time = break_time
        
        self._setup_ui()
    
    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(tk.Tk(), cfg.get("break_time", 15))
    
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
        
    def show_break_popup(self, on_break_end: Optional[callable] = None):
        if self.popup is not None:
            return
        self.remaining_break_time = self.break_time
        self.popup = tk.Toplevel(self.root)
        self.popup.title("Take a break :)")
        self.popup.geometry("300x100")
        self.timer_label = tk.Label(self.popup, text=f"Next Capture in: {self.remaining_break_time} s", font=("Arial", 18))
        self.timer_label.pack(expand=True)
        self.popup_timer = self.root.after(1000, lambda: self.update_popup_timer(on_break_end))
    
    def update_popup_timer(self, on_break_end: Optional[callable] = None):
        if self.remaining_break_time > 0:
            self.timer_label.config(text=f"Next Capture in: {self.remaining_break_time} s")
            self.remaining_break_time -= 1
            self.popup_timer = self.root.after(1000, lambda: self.update_popup_timer(on_break_end))
        else:
            self.close_break_popup(on_break_end)
    
    def close_break_popup(self, on_break_end: Optional[callable] = None):
        if self.popup is None:
            return
        self.popup.destroy()
        self.popup = None
        if on_break_end:
            on_break_end()


class DataCollectorDelegate:
    pass