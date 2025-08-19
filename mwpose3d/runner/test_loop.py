from pathlib import Path
from typing import TYPE_CHECKING, Dict, Sequence, Union
import torch
from torch.utils.data import DataLoader

from mmengine.runner.amp import autocast
from mwpose3d.evaluation.metrics.base import BaseMetric
from mwpose3d.evaluation.postprocessing.base import ComposePostProcess
from mwpose3d.registry import METRICS, LOOPS
from .base_loop import BaseLoop

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
        postprocess: list = None,
        fp16: bool = False,
    ):
        super().__init__(runner, dataloader)
        self._iter = 0
        self.report = []
        self.gt_data = []  # Collect all ground truth data
        self.pred_data = []  # Collect all prediction data
        
        self.last_output = dict()
        
        self.postprocess = ComposePostProcess(postprocess) if postprocess is not None else None
        if self.postprocess is not None:
            print("Postprocessing is enabled.")
        self.evaluator: BaseMetric = METRICS.build(metric_cfg)
        self.checkpoints = None #checkpoints
        self.stop_testing = False
        self.fp16 = fp16
        
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
            self.runner.call_hook('after_test_epoch')

        else:
            load_from_root = Path(self.runner._load_from).parent
            for checkpoint in self.checkpoints:
                checkpoint_path = load_from_root / f"epoch_{checkpoint}.pth"
                self.runner.load_checkpoint(str(checkpoint_path))
                self.runner.call_hook('before_test_epoch')
                self._run_epoch()
                summary = self.evaluator.evaluate()
                self.runner.call_hook('after_test_epoch')

        self.runner.call_hook('after_test')
        return self.runner.model
        
    def _run_epoch(self) -> None:
        self.runner.model.eval()
        with torch.no_grad():
            for idx, data_batch in tqdm(enumerate(self.dataloader), 
                          total=len(self.dataloader),
                          desc='Testing'):
                self._run_iter(idx, data_batch)
                if self.stop_testing:
                    break

    @torch.no_grad()
    def _run_iter(self, idx: int, data_batch: dict) -> None:
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        assert hasattr(self.runner.model, 'pack_input')
        batch_inputs, data_samples = self.runner.model.pack_input(dict(data_batch, previous_output=self.last_output))
        assert len(data_samples) == 1, 'TestLoop only supports batch_size=1'
        with autocast(enabled=self.fp16):
            outputs = self.runner.model(batch_inputs, data_samples, mode='predict')
        self.last_output = outputs
        data_samples_0 = data_samples[0]
        if self.postprocess is not None:
            batch_inputs, data_samples_0 = self.postprocess(batch_inputs, data_samples_0)
        self.evaluator.process_sample(data_samples_0, data_batch=data_batch)
        
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

