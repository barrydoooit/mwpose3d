from typing import TYPE_CHECKING, Dict, List, Sequence, Union
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mwpose3d.datasets.skel_data_sample import SkeletonDataSample
from mwpose3d.evaluation.metrics.base import BaseMetric
from mwpose3d.registry import METRICS, LOOPS
from .base_loop import BaseLoop



if TYPE_CHECKING:
    from mwpose3d.runner.runner import Runner

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

        self.last_output = dict()
        
        self.evaluator: BaseMetric = METRICS.build(metric_cfg)
        
    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self._run_epoch()
        summary = self.evaluator.evaluate()
        self.evaluator.reset()
        self.runner.call_hook('after_val_epoch', metrics=summary)
        self.runner.call_hook('after_val')
        
        return self.runner.model
    
    def _run_epoch(self) -> None:
        self.runner.model.eval()
        with torch.no_grad():
            for idx, data_batch in tqdm(enumerate(self.dataloader), 
                          total=len(self.dataloader),
                          desc='Validating'):
                self._run_iter(idx, data_batch)

        
    def _run_iter(self, idx: int, data_batch: dict) -> None:
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        assert hasattr(self.runner.model, 'pack_input')
        batch_inputs, data_samples = self.runner.model.pack_input(dict(data_batch, previous_output=self.last_output), training=False)
        assert len(data_samples) == 1, 'TestLoop only supports batch_size=1'
        outputs = self.runner.model(batch_inputs, data_samples, mode='predict')
        self.last_output = outputs
        self.evaluator.process_sample(data_samples[0], data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
                
        self._iter += 1

