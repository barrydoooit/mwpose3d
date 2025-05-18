from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence, Union
import torch
from torch.utils.data import DataLoader

from mwpose3d.datasets.skel_data_sample import SkeletonDataSample
from mwpose3d.evaluation.metrics.base import BaseMetric
from mwpose3d.registry import METRICS, LOOPS
from .base_loop import BaseLoop

import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

if TYPE_CHECKING:
    from mwpose3d.runner.runner import Runner
from tqdm import tqdm

@LOOPS.register_module()
class TestLoop(BaseLoop):
    def __init__(
        self,
        runner: 'Runner',
        dataloader: Union[DataLoader, Dict],
        metric_cfg: dict,
        checkpoints: Sequence[str] = None,
    ):
        super().__init__(runner, dataloader)
        self._iter = 0
        self.report = []
        self.gt_data = []  # Collect all ground truth data
        self.pred_data = []  # Collect all prediction data
        
        self.last_output = dict()
        
        self.evaluator: BaseMetric = METRICS.build(metric_cfg)
        self.checkpoints = checkpoints
        
    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        self.runner.call_hook('before_test')
        if self.checkpoints is None:
            self.runner.call_hook('before_test_epoch')
            self._run_epoch()
            summary = self.evaluator.evaluate()
            self.runner.call_hook('after_test_epoch', metrics=summary)

        else:
            load_from_root = Path(self.runner._load_from).parent
            for checkpoint in self.checkpoints:
                checkpoint_path = load_from_root / f"epoch_{checkpoint}.pth"
                self.runner.load_checkpoint(str(checkpoint_path))
                self.runner.call_hook('before_test_epoch')
                self._run_epoch()
                summary = self.evaluator.evaluate()
                self.runner.call_hook('after_test_epoch', metrics=summary)

        self.runner.call_hook('after_test')
        return self.runner.model
        
    def _run_epoch(self) -> None:
        self.runner.model.eval()
        with torch.no_grad():
            for idx, data_batch in tqdm(enumerate(self.dataloader), 
                          total=len(self.dataloader),
                          desc='Testing'):
                self._run_iter(idx, data_batch)

    def _run_iter(self, idx: int, data_batch: dict) -> None:
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        assert hasattr(self.runner.model, 'pack_input')
        batch_inputs, data_samples = self.runner.model.pack_input(dict(data_batch, previous_output=self.last_output))
        assert len(data_samples) == 1, 'TestLoop only supports batch_size=1'
        outputs = self.runner.model(batch_inputs, data_samples, mode='predict')
        self.last_output = outputs
        self.evaluator.process_sample(data_samples[0], data_batch=data_batch)
        
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

