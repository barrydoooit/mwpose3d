import tkinter as tk
from tkinter import ttk

import pandas as pd



class CheckList(ttk.LabelFrame):
    def __init__(self, parent, title, callback):
        super().__init__(parent, text=title)
        self.callback = callback
        self.checked_items = []
        
        # 滚动区域
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview)
        self.inner_frame = ttk.Frame(self.canvas)
        
        # 布局设置
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # 绑定滚动事件
        self.inner_frame.bind("<Configure>", self.on_configure)
        self.canvas.create_window((0,0), window=self.inner_frame, anchor="nw")

        # 存储复选框状态
        self.vars = {}
        self.checkboxes = {}

    def on_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def set_items(self, items):
        # 清除旧内容
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.vars.clear()
        self.checkboxes.clear()
        
        # 创建新的复选框
        for item in sorted(items):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(
                self.inner_frame, 
                text=item, 
                variable=var,
                command=lambda v=var, i=item: self.update_checked(v, i)
            )
            cb.pack(anchor="w")
            self.vars[item] = var
            self.checkboxes[item] = cb

    def update_checked(self, var, item):
        if var.get():
            self.checked_items.append(item)
        else:
            self.checked_items.remove(item)
        self.callback()

class MultiColumnCheckList(ttk.LabelFrame):
    def __init__(self, parent, title, callback, columns):
        """
        :param columns: List of additional column headers (operations) to display.
        """
        super().__init__(parent, text=title)
        self.callback = callback
        self.checked_items = []  # List of selected episode names.
        self.columns = columns   # Operation columns.
        self.vars = {}  # Mapping: episode -> BooleanVar for selection.
        self.cells = {}  # Mapping: episode -> {column_name: label widget}

        # Create a scrollable area.
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner_frame = ttk.Frame(self.canvas)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.inner_frame.bind("<Configure>", self.on_configure)

    def on_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def set_items(self, df: pd.DataFrame):
        """
        Expects a DataFrame with episode names as the index and each column representing an operation.
        Cells should contain booleans (True indicates the operation has been done).
        """
        # Clear previous widgets.
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.vars.clear()
        self.cells.clear()
        self.checked_items = []

        # Create header row.
        header_font = ("TkDefaultFont", 10, "bold")
        tk.Label(self.inner_frame, text="Episode", font=header_font, borderwidth=1,
                 relief="solid", padx=5, pady=2).grid(row=0, column=0, sticky="nsew")
        for col_idx, col_name in enumerate(self.columns, start=1):
            tk.Label(self.inner_frame, text=col_name, font=header_font, borderwidth=1,
                     relief="solid", padx=5, pady=2).grid(row=0, column=col_idx, sticky="nsew")

        # Create rows for each episode.
        sorted_episodes = sorted(df.index)
        for row_idx, episode in enumerate(sorted_episodes, start=1):
            # Create a checkbutton for selecting the episode.
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(
                self.inner_frame, text=episode, variable=var,
                command=lambda ep=episode, v=var: self.update_checked(ep, v)
            )
            cb.grid(row=row_idx, column=0, sticky="nsew", padx=1, pady=1)
            self.vars[episode] = var

            # Create cells for each operation.
            self.cells[episode] = {}
            for col_idx, col_name in enumerate(self.columns, start=1):
                # Display a tick if the cell is True.
                value = df.at[episode, col_name]
                display = "Yes" if value else "No"
                label = tk.Label(self.inner_frame, text=display, borderwidth=1,
                                 relief="solid", padx=5, pady=2)
                label.grid(row=row_idx, column=col_idx, sticky="nsew", padx=1, pady=1)
                self.cells[episode][col_name] = label

        # Configure grid weights.
        total_cols = len(self.columns) + 1
        for col in range(total_cols):
            self.inner_frame.grid_columnconfigure(col, weight=1)

    def update_checked(self, episode, var):
        if var.get():
            if episode not in self.checked_items:
                self.checked_items.append(episode)
        else:
            if episode in self.checked_items:
                self.checked_items.remove(episode)
        if self.callback:
            self.callback()