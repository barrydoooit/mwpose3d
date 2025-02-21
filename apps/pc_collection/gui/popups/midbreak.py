import tkinter as tk
from typing import Optional, Callable

class MidBreakPopup(tk.Toplevel):
    def __init__(self,
                 master: tk.Tk,
                 break_time: int,
                 geometry: str = "300x100",
                 on_break_end: Optional[Callable] = None):
        super().__init__(master)
        self.break_time = break_time
        self.remaining_break_time = break_time
        self.on_break_end = on_break_end
        
        self.title("Before Capture Starts")
        self.geometry(geometry)
        
        self.timer_label = tk.Label(self,
                                    text=f"Next Capture in: {self.remaining_break_time} s",
                                    font=("Arial", 18))
        self.timer_label.pack(expand=True)
        
        self.popup_timer = self.after(1000, self.update_popup_timer)
    
    def update_popup_timer(self):
        if self.remaining_break_time > 0:
            self.timer_label.config(text=f"Next Capture in: {self.remaining_break_time} s")
            self.remaining_break_time -= 1
            self.popup_timer = self.after(1000, self.update_popup_timer)
        else:
            self.popup_timer = None
            self.close_break_popup()
    
    def close_break_popup(self, with_callback: bool = True):
        self.destroy()
        if self.popup_timer is not None:
            try:
                self.after_cancel(self.popup_timer)
            except Exception:
                pass
            self.popup_timer = None
        if not with_callback:
            return
        if self.on_break_end:
            self.on_break_end()