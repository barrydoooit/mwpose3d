
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Set

import pandas as pd


class CheckList(ttk.LabelFrame):
    """(Kept for backward compatibility) Simple single-column checklist with scrolling."""
    def __init__(self, parent, title, callback):
        super().__init__(parent, text=title)
        self.callback = callback
        self.checked_items: List[str] = []

        # Scrollable area
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner_frame = ttk.Frame(self.canvas)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind scrolling
        self.inner_frame.bind("<Configure>", self.on_configure)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        # Mouse-wheel (scroll where hovered)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        self.vars: Dict[str, tk.BooleanVar] = {}
        self.checkboxes: Dict[str, ttk.Checkbutton] = {}

    # --- mouse wheel helpers ---
    def _on_mousewheel(self, event):
        if event.num == 4:     # Linux scroll up
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:   # Linux scroll down
            self.canvas.yview_scroll(3, "units")
        else:                  # Windows / macOS
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_mousewheel(self, _):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    # --- content ---
    def on_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def set_items(self, items: List[str]):
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.vars.clear()
        self.checkboxes.clear()

        for item in sorted(items):
            var = tk.BooleanVar(value=item in self.checked_items)
            cb = ttk.Checkbutton(
                self.inner_frame,
                text=item,
                variable=var,
                command=lambda v=var, i=item: self.update_checked(v, i),
            )
            cb.pack(anchor="w")
            self.vars[item] = var
            self.checkboxes[item] = cb

    def update_checked(self, var: tk.BooleanVar, item: str):
        if var.get():
            if item not in self.checked_items:
                self.checked_items.append(item)
        else:
            if item in self.checked_items:
                self.checked_items.remove(item)
        if self.callback:
            self.callback()


class MultiColumnCheckList(ttk.LabelFrame):
    """
    A multi-column, scrollable list of episodes with checkboxes per row and
    clickable headers for sorting.

    New in this reimplementation:
    - Shift-click range selection (selects all rows between last anchor and clicked row).
    - Mouse wheel scrolling on hover.
    - Clickable headers to sort by any column (including Episode). Clicking again toggles
      the order; switching column resets to ascending. Tie-breaker is always Episode ascending.
    - Displays arbitrary per-episode key/value columns (e.g., from _pcd_meta) and optional
      boolean operation columns (e.g., Aligned). Missing values render as '-'.
    """
    def __init__(self, parent, title, callback, columns: Optional[List[str]] = None):
        super().__init__(parent, text=title)
        self.callback = callback
        self.checked_items: List[str] = []                  # episode names selected
        self.columns: List[str] = columns or []             # display columns (excluding "Episode")
        self.bool_columns: Set[str] = set()                 # subset of columns whose values are booleans
        self.vars: Dict[str, tk.BooleanVar] = {}            # episode -> BooleanVar
        self.cells: Dict[str, Dict[str, tk.Widget]] = {}    # episode -> {col_name: label widget}
        self.df: Optional[pd.DataFrame] = None              # backing DataFrame
        self._row_order: List[str] = []                     # current visual order of episode names
        self._last_anchor_index: Optional[int] = None       # for shift range selection
        self._sort_column: str = "Episode"                  # current sort column ("Episode" or a column in df)
        self._ascending: bool = True                        # current sort order

        # Scrollable area
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner_frame = ttk.Frame(self.canvas)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.inner_frame.bind("<Configure>", self._on_configure)

        # Mouse wheel (scroll where hovered)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    # --- scrolling ---
    def _on_mousewheel(self, event):
        if event.num == 4:     # Linux scroll up
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:   # Linux scroll down
            self.canvas.yview_scroll(3, "units")
        else:                  # Windows / macOS
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_mousewheel(self, _):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # --- building / refreshing ---
    def set_items(self, df: pd.DataFrame, columns: Optional[List[str]] = None):
        """
        Provide a DataFrame with episode names as index and columns to display.
        The 'columns' argument (if provided) selects/arranges the visible columns (besides the 'Episode' column).
        Values will be displayed as strings, except boolean columns which render as 'Yes' / 'No'.
        """
        self.df = df.copy()
        # normalize dataframe values: fillna('-') for display columns; BUT keep original booleans
        self.df = self.df.copy()
        # Detect booleans BEFORE filling, then fill the rest with '-'
        self.bool_columns = {c for c in self.df.columns if str(self.df[c].dtype) in {"bool", "boolean"}}
        self.df = self.df.fillna("-")

        # Display columns (besides the 'Episode' checkbox column)
        if columns is not None:
            self.columns = list(columns)
        else:
            self.columns = list(self.df.columns)

        # Clear previous UI
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.vars.clear()
        self.cells.clear()

        # Prepare initial order (respect current sort settings)
        all_eps = list(self.df.index)
        self._row_order = self._sorted_episodes(all_eps)

        # Header row: "Episode" + each column as a clickable button
        header_font = ("TkDefaultFont", 10, "bold")

        def header_btn_text(name: str) -> str:
            arrow = ""
            if self._sort_column == name:
                arrow = " ▲" if self._ascending else " ▼"
            return f"{name}{arrow}"

        ep_btn = ttk.Button(self.inner_frame, text=header_btn_text("Episode"),
                            command=lambda: self._sort_by("Episode"))
        ep_btn.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        for col_idx, col_name in enumerate(self.columns, start=1):
            btn = ttk.Button(self.inner_frame, text=header_btn_text(col_name),
                             command=lambda c=col_name: self._sort_by(c))
            btn.grid(row=0, column=col_idx, sticky="nsew", padx=1, pady=1)

        # Build rows
        for row_idx, episode in enumerate(self._row_order, start=1):
            # checkbox
            var = tk.BooleanVar(value=(episode in self.checked_items))
            cb = ttk.Checkbutton(self.inner_frame, text=episode, variable=var)
            cb.grid(row=row_idx, column=0, sticky="nsew", padx=1, pady=1)

            # Bind clicks for shift-range selection & normal toggling
            cb.bind("<Button-1>", lambda e, idx=row_idx - 1, ep=episode: self._on_row_click(e, idx, ep))

            self.vars[episode] = var
            self.cells[episode] = {}

            # data cells
            for col_idx, col_name in enumerate(self.columns, start=1):
                value = self.df.at[episode, col_name] if col_name in self.df.columns else "-"
                if col_name in self.bool_columns:
                    display = "Yes" if bool(value) else "No"
                else:
                    display = "-" if value is None else str(value)
                lbl = tk.Label(self.inner_frame, text=display, borderwidth=1, relief="solid", padx=5, pady=2)
                lbl.grid(row=row_idx, column=col_idx, sticky="nsew", padx=1, pady=1)
                self.cells[episode][col_name] = lbl

        # Configure grid weights
        total_cols = len(self.columns) + 1
        for col in range(total_cols):
            self.inner_frame.grid_columnconfigure(col, weight=1)

    # --- selection ---
    def _on_row_click(self, event, idx: int, episode: str):
        """
        Handle click on a row's checkbox.
        - Normal click toggles just that row.
        - Shift-click selects a contiguous range from last anchor to this row (all set to True).
        """
        shift_held = False
        try:
            # event.state bit 0x0001 is usually the Shift mask on Tk
            shift_held = bool(event.state & 0x0001)
        except Exception:
            shift_held = False

        # Prevent the default Checkbutton toggle; we will manage the state manually.
        event.widget.after_idle(lambda: None)
        # Returning "break" prevents the default widget handling of the event
        # so we ensure our custom behavior is the only change.
        result = "break"

        if shift_held and self._last_anchor_index is not None and 0 <= self._last_anchor_index < len(self._row_order):
            start = min(self._last_anchor_index, idx)
            end = max(self._last_anchor_index, idx)
            for i in range(start, end + 1):
                ep = self._row_order[i]
                if not self.vars[ep].get():
                    self.vars[ep].set(True)
                    if ep not in self.checked_items:
                        self.checked_items.append(ep)
            # Keep anchor (common desktop behavior)
        else:
            # toggle single
            new_val = not self.vars[episode].get()
            self.vars[episode].set(new_val)
            if new_val:
                if episode not in self.checked_items:
                    self.checked_items.append(episode)
            else:
                if episode in self.checked_items:
                    self.checked_items.remove(episode)
            # Set new anchor
            self._last_anchor_index = idx

        if self.callback:
            self.callback()
        return result

    # --- sorting ---
    def _sort_by(self, column: str):
        if self._sort_column == column:
            # Toggle direction
            self._ascending = not self._ascending
        else:
            self._sort_column = column
            self._ascending = True  # reset when switching column

        if self.df is None:
            return

        # Recompute order and rebuild only rows + header arrows
        self._row_order = self._sorted_episodes(list(self.df.index))

        # Rebuild everything to simplify (preserving checked state)
        current_checked = set(self.checked_items)
        self.checked_items = list(current_checked)  # keep as list
        self.set_items(self.df, columns=self.columns)

    def _coerce_for_sort(self, value):
        if value is None or value == "-":
            return ""  # empty string sorts first on ascending, last on descending when reversed
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, bool):
            return value
        # try numeric
        s = str(value)
        try:
            return float(s)
        except Exception:
            return s.lower()

    def _sorted_episodes(self, episodes: List[str]) -> List[str]:
        if self.df is None:
            return sorted(episodes)

        # stable two-pass sort to ensure tie-breaker by Episode ascending always
        # 1) base: Episode ascending
        ordered = sorted(episodes)

        # 2) primary: chosen column
        if self._sort_column == "Episode":
            if not self._ascending:
                ordered = list(reversed(ordered))
            # else already ascending
            return ordered

        col = self._sort_column
        def primary(ep: str):
            val = self.df.at[ep, col] if col in self.df.columns else None
            return self._coerce_for_sort(val)

        ordered = sorted(ordered, key=primary, reverse=not self._ascending)
        return ordered
