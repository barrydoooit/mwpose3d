from typing import TYPE_CHECKING, Dict, List, Sequence, Union
import torch
from torch.utils.data import DataLoader

from dl_engine.dataset.skel_data_sample import SkeletonDataSample
from dl_engine.runner.evaluate.base import METRICS, BaseMetric
from kinect_toolkits.kinectData import KeypointType, Connectivity
from .base_loop import BaseLoop, LOOPS

import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

if TYPE_CHECKING:
    from dl_engine.runner.runner import Runner

@LOOPS.register_module()
class ValLoop(BaseLoop):
    def __init__(
        self,
        runner: 'Runner',
        dataloader: Union[DataLoader, Dict],
        metric_cfg: dict
    ):
        super().__init__(runner, dataloader)
        self._iter = 0
        self.report = []
        self.gt_data = []  # Collect all ground truth data
        self.pred_data = []  # Collect all prediction data
        
        self.evaluator: BaseMetric = METRICS.build(metric_cfg)
        
    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        self._run_epoch()
        summary = self.evaluator.evaluate()
        self.evaluator.reset()
        return self.runner.model
    
    def _run_epoch(self) -> None:
        self.runner.model.eval()
        with torch.no_grad():
            for idx, data_batch in enumerate(self.dataloader):
                self._run_iter(idx, data_batch)
    
    def _run_iter(self, idx: int, data_batch: dict) -> None:
        assert hasattr(self.runner.model, 'pack_input')
        batch_inputs, data_samples = self.runner.model.pack_input(data_batch)
        assert len(data_samples) == 1, 'TestLoop only supports batch_size=1'
        _ = self.runner.model(batch_inputs, data_samples, mode='predict')

        self.evaluator.process_sample(data_samples[0])
        self._iter += 1

