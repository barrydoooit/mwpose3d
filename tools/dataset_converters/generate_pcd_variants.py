"""
generate_pcd_variants.py
------------------------
Re-processes raw ADC .bin files using one or more alternative DSP configs to
produce additional point cloud variants in an existing formatted dataset.

Usage (headless)
----------------
gen = PcdVariantGenerator(
    dataset='20260323barrytrail',
    data_dir=Path('data/20260323barrytrail'),
    raw_traces_dir=Path('win_workspace/.../traces/20260323barrytrail'),
    dsp_cfg_paths=[Path('configs/dsp/ti-mobile-tracker-64loops_xWR1843.py')],
)
gen.generate()

Usage (GUI — user selects configs interactively)
-------------------------------------------------
gen = PcdVariantGenerator(
    dataset='20260323barrytrail',
    data_dir=Path('data/20260323barrytrail'),
    raw_traces_dir=Path('win_workspace/.../traces/20260323barrytrail'),
    dsp_cfg_dir=Path('configs/dsp'),         # directory pre-loaded into checklist
)
gen.select_configs_and_generate()            # opens Tkinter dialog, then processes
"""

import os
from pathlib import Path
from typing import List, Optional


class PcdVariantGenerator:
    """
    Orchestrates multi-DSP-config point cloud generation for a dataset.

    Either supply *dsp_cfg_paths* directly (headless) or call
    select_configs_and_generate() to pick configs via a GUI (GUI mode).
    """

    def __init__(self,
                 dataset: str,
                 data_dir: Path,
                 raw_traces_dir: Path,
                 dsp_cfg_paths: Optional[List[Path]] = None,
                 dsp_cfg_dir: Optional[Path] = None,
                 episode_ids: Optional[List[str]] = None):
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        self.raw_traces_dir = Path(raw_traces_dir)
        self.dsp_cfg_paths = [Path(p) for p in dsp_cfg_paths] if dsp_cfg_paths else []
        self.dsp_cfg_dir = Path(dsp_cfg_dir) if dsp_cfg_dir else None
        self.episode_ids = episode_ids

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def generate(self):
        """Run reprocessing for the pre-configured DSP configs (headless)."""
        if not self.dsp_cfg_paths:
            raise ValueError(
                'No DSP config paths provided. '
                'Pass dsp_cfg_paths= or call select_configs_and_generate().'
            )
        self._run(self.dsp_cfg_paths)

    def select_configs_and_generate(self):
        """
        Open a GUI dialog, let the user tick DSP config files, then generate.

        If dsp_cfg_dir is not set, falls back to the default configs/dsp/
        directory relative to this file.
        """
        default_dir = self.dsp_cfg_dir or (
            Path(__file__).parent.parent.parent / 'configs' / 'dsp'
        )
        selected = _select_dsp_configs_gui(str(default_dir))
        if not selected:
            print('[PcdVariantGenerator] No configs selected.')
            return
        self._run([Path(p) for p in selected])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, cfg_paths: List[Path]):
        from tools.rawproc.pcd_reprocessor import PointCloudReprocessor

        for cfg_path in cfg_paths:
            print(f'\n[PcdVariantGenerator] Applying DSP config: {cfg_path.name}')
            reprocessor = PointCloudReprocessor(
                formatted_dataset_dir=self.data_dir,
                raw_traces_dir=self.raw_traces_dir,
                dsp_cfg_path=cfg_path,
            )
            reprocessor.process_all(episode_ids=self.episode_ids)


# ---------------------------------------------------------------------------
# GUI helper
# ---------------------------------------------------------------------------

def _select_dsp_configs_gui(default_dir: str) -> List[str]:
    """
    Open a Tkinter checklist dialog for the user to select DSP config .py
    files from a directory.

    Returns a list of absolute file paths for the selected configs.
    """
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    selected_paths: List[str] = []

    root = tk.Tk()
    root.title('Select DSP Configs')
    root.geometry('550x420')
    root.resizable(True, True)

    dir_var = tk.StringVar(value=os.path.abspath(default_dir))
    check_vars: dict = {}

    def browse_dir():
        chosen = filedialog.askdirectory(
            title='Select DSP config directory',
            initialdir=dir_var.get(),
        )
        if chosen:
            dir_var.set(chosen)
            _refresh_list()

    def _refresh_list():
        for widget in list_frame.winfo_children():
            widget.destroy()
        check_vars.clear()
        d = dir_var.get()
        if not os.path.isdir(d):
            return
        py_files = sorted(f for f in os.listdir(d) if f.endswith('.py'))
        for fname in py_files:
            var = tk.BooleanVar(value=False)
            check_vars[fname] = var
            ttk.Checkbutton(list_frame, text=fname, variable=var).pack(
                anchor='w', padx=8, pady=2
            )

    dir_frame = ttk.Frame(root)
    dir_frame.pack(fill='x', padx=10, pady=(10, 0))
    ttk.Label(dir_frame, text='Config directory:').pack(side='left')
    ttk.Entry(dir_frame, textvariable=dir_var, width=38).pack(side='left', padx=4)
    ttk.Button(dir_frame, text='Browse…', command=browse_dir).pack(side='left')

    ttk.Label(root, text='Select DSP config files to apply:').pack(
        anchor='w', padx=10, pady=(8, 0)
    )

    canvas = tk.Canvas(root, borderwidth=0)
    scrollbar = ttk.Scrollbar(root, orient='vertical', command=canvas.yview)
    list_frame = ttk.Frame(canvas)
    list_frame.bind(
        '<Configure>',
        lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
    )
    canvas.create_window((0, 0), window=list_frame, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side='left', fill='both', expand=True, padx=10, pady=4)
    scrollbar.pack(side='right', fill='y', pady=4)

    _refresh_list()

    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill='x', padx=10, pady=8)

    def on_confirm():
        d = dir_var.get()
        for fname, var in check_vars.items():
            if var.get():
                selected_paths.append(os.path.join(d, fname))
        if not selected_paths:
            messagebox.showwarning('No selection', 'Please tick at least one config file.')
            return
        root.destroy()

    def on_cancel():
        root.destroy()

    ttk.Button(btn_frame, text='Run', command=on_confirm).pack(side='right', padx=4)
    ttk.Button(btn_frame, text='Cancel', command=on_cancel).pack(side='right')

    root.mainloop()
    return selected_paths
