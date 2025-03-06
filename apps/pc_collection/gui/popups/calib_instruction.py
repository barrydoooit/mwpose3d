from enum import Enum
import tkinter as tk
import json
from typing import Any, Optional, Callable, TYPE_CHECKING, Tuple
import threading

if TYPE_CHECKING:
    from apps.common.pcd.pc_buffer import PointCloudBuffer



class TimeCalibInstructionPopup(tk.Toplevel):
    class Stages(Enum):
        ENTRY_STAGE: int = 0
        MOVE_STAGE: int = 1
        TRANSIT_STAGE: int = 2
        STILL_STAGE: int = 3
        CLOSE_STAGE: int = 4
    
    def __init__(self,
                 master: tk.Tk,
                 pcd_buffer: "PointCloudBuffer",
                 stages_duration: Tuple[int, int, int, int],
                 ticks_to_confirm: int,
                 geometry: str = "400x200",
                 before_popup_close: Optional[Callable[["TimeCalibInstructionPopup"], Any]] = None,
                 ):
        super().__init__(master)
        self.pcd_buffer = pcd_buffer
        self.move_duration = stages_duration[0]
        self.transit_duration = stages_duration[1]
        self.still_duration = stages_duration[2]
        self.close_lag = stages_duration[3]
        self.ticks_to_confirm = ticks_to_confirm
        
        self.still_start_ts = None
        self.still_end_ts = None
        self.before_popup_close = before_popup_close
        
        self.instruction_label = tk.Label(self, text="", font=("Arial", 14))
        self.instruction_label.pack(pady=10)
        self.timer_label = tk.Label(self, text="", font=("Arial", 16))
        self.timer_label.pack(pady=10)
        
        self.timer_id = None
        self.remaining_time = 0
        self.current_stage: 'TimeCalibInstructionPopup.Stages' = self.Stages.ENTRY_STAGE
        
        self.title("Time Calibration")
        self.geometry(geometry)
        
        self.start_move_stage()
        
    def start_move_stage(self):
        self.current_stage = self.Stages.MOVE_STAGE
        self.instruction_label.config(text="Move your right arm up and down and do not stop.")
        self.remaining_time = self.move_duration
        self.update_timer_label()
        self.schedule_timer(self.move_stage_tick)
    
    def move_stage_tick(self):
        print(f"Pcd buffer entrance threshold: {self.pcd_buffer._frame_entrance_threshold}")
        
        self.remaining_time -= 1
        if self.remaining_time <= 0:
            self.start_transition_stage()
        else:
            self.update_timer_label()
            self.schedule_timer(self.move_stage_tick)
    
    def start_transition_stage(self):
        self.current_stage = self.Stages.TRANSIT_STAGE
        self.instruction_label.config(text="Stop the movements.")
        self.pcd_buffer.change_frame_entrance_threshold(8)
        with self.pcd_buffer.locked_buffer() as buffer:
            self.initial_frame_count = len(buffer)
            self.still_start_ts = buffer[-1].ts
        self.remaining_time = self.transit_duration
        self.update_timer_label()
        self.schedule_timer(lambda: self.transition_tick(self.ticks_to_confirm), interval=500)
    
    def transition_tick(self, ticks_to_confirm):
        with self.pcd_buffer.locked_buffer() as buffer:
            current_frame_count = len(buffer)
        if current_frame_count == self.initial_frame_count:
            ticks_to_confirm -= 1
            print("Ticks remaining:", ticks_to_confirm)
            if ticks_to_confirm <= 0:
                self.remaining_time = 0
                self.update_timer_label()
                if self.timer_id is not None:
                    self.after_cancel(self.timer_id)
                    self.timer_id = None
                self.start_still_stage()
                return
        else:
            if ticks_to_confirm < self.ticks_to_confirm:
                self.instruction_label.config(text="Movement not stopped! Restarting process...")
                self.after(1000, self.restart_process)
                return
            ticks_to_confirm = self.ticks_to_confirm
            self.initial_frame_count = current_frame_count
            self.still_start_ts = buffer[-1].ts
        
        self.remaining_time -= 1
        if self.remaining_time <= 0:
            self.instruction_label.config(text="Movement not stopped! Restarting process...")
            self.after(1000, self.restart_process)
            return
        
        self.update_timer_label()
        self.schedule_timer(lambda: self.transition_tick(ticks_to_confirm), interval=500)
        
    def start_still_stage(self):
        self.current_stage = self.Stages.STILL_STAGE
        self.instruction_label.config(text="Stay still......")
        with self.pcd_buffer.locked_buffer() as buffer:
            self.initial_frame_count = len(buffer)
        self.remaining_time = self.still_duration
        self.update_timer_label()
        self.schedule_timer(self.still_stage_tick)
    
    def still_stage_tick(self):
        with self.pcd_buffer.locked_buffer() as buffer:
            current_frame_count = len(buffer)
        if current_frame_count > self.initial_frame_count:
            self.instruction_label.config(text="Movement detected! Restarting process...")
            self.after(1000, self.restart_process)
            return
        
        self.remaining_time -= 1
        if self.remaining_time <= 0:
            # with self.pcd_buffer.locked_buffer() as buffer:
            #     self.still_end_ts = buffer[-1].ts
            self.start_close_stage()
            return
        
        self.update_timer_label()
        self.schedule_timer(self.still_stage_tick)
    
    def start_close_stage(self):
        self.current_stage = self.Stages.CLOSE_STAGE
        self.instruction_label.config(text="Calibration Success! Closing the process...")
        self.remaining_time = self.close_lag
        self.update_timer_label()
        self.schedule_timer(self.close_stage_tick)
    
    def close_stage_tick(self):
        self.remaining_time -= 1
        if self.remaining_time <= 0:
            self.close_popup()
            return
        self.update_timer_label()
        self.schedule_timer(self.close_stage_tick)


    def restart_process(self):
        self.current_stage = self.Stages.ENTRY_STAGE
        self.pcd_buffer.clear()
        self.pcd_buffer.change_frame_entrance_threshold(0)
        print("Buffer cleared")
        self.start_move_stage()

    def update_timer_label(self):
        self.timer_label.config(text=f"Time remaining: {self.remaining_time} s")

    def schedule_timer(self, callback, interval=1000):
        """Cancel any previous timer and schedule a new one after 1000 ms."""
        if self.timer_id is not None:
            self.after_cancel(self.timer_id)
        self.timer_id = self.after(interval, callback)
    
    def close_popup(self, with_callback: bool = True):
        self.pcd_buffer.change_frame_entrance_threshold(0)
        if not with_callback:
            self.destroy()
            return
        # self.still_period = (self.still_start_ts, self.still_end_ts,)
        self.still_period = (self.still_start_ts, self.still_start_ts + self.still_duration * 1000,)
        if self.before_popup_close is not None:
            self.before_popup_close(self)
        self.destroy()
        