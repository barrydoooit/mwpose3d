from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List, Literal, Sequence, Union
import torch
from torch.utils.data import DataLoader


from .utils import AverageMeter
from dl_engine.runner.evaluate.base import METRICS, BaseMetric
from dl_engine.runner.base_loop import BaseLoop, LOOPS

if TYPE_CHECKING:
    from dl_engine.runner.runner import Runner



@LOOPS.register_module()
class MMDiffValLoop(BaseLoop):
    def __init__(
        self,
        runner: 'Runner',
        dataloader: Union[DataLoader, Dict],
        metric_cfg: dict
    ):
        super().__init__(runner, dataloader)
        self._iter = 0
        self.evaluator: BaseMetric = METRICS.build(metric_cfg)
        
        
    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        self.hook_before_val()
        self._run_epoch()
        summary = self.evaluator.evaluate()
        self.evaluator.reset()
        return self.runner.model
    
    def _run_epoch(self) -> None:
        self.hook_before_val_epoch()
        self.runner.model.eval()
        with torch.no_grad():
            for idx, data_batch in enumerate(self.dataloader):
                self._run_iter(idx, data_batch)
    
    def _run_iter(self, idx: int, data_batch: dict) -> None:
        assert hasattr(self.runner.model, 'pack_input')
        batch_inputs, data_samples = self.runner.model.pack_input(data_batch)
        assert len(data_samples) == 1, 'TestLoop only supports batch_size=1'
        _ = self.runner.model(batch_inputs, data_samples, mode='predict')

        self.evaluator.process_sample(data_samples[-1], data_batch=data_batch)
        self._iter += 1

    # def hook_before_val(self):
    #     """Hook before validation."""
    #     self.phase: Literal[1, 2] = 1
    #     cudnn.benchmark = True
    
    # def hook_before_val_epoch(self):
    #     train_loop = self.runner.train_loop
    #     val_loop = self.runner.val_loop
    #     self = self.runner
    #     phase = 1 if train_loop.epoch <= train_loop.pretrain_max_epochs else 2 
    #     if phase == 1:
    #         self.model.mode = 'pred_coarse'
    #     elif phase == 2:
    #         self.model.mode = 'pred_fine'
    #     else:
    #         raise ValueError(f"Invalid phase: {phase}. Only 1 and 2 are supported.")
        
    #     # if train_loop.epoch == train_loop.pretrain_max_epochs:
    #     #     new_num_frames = train_loop.phase2_cfg.get('num_frames')
    #     #     new_dataloader_cfg = deepcopy(val_loop.dataloader_cfg)
    #     #     for transform in new_dataloader_cfg['dataset']['pipeline']:
    #     #         if transform['type'] == 'SequenceClip':
    #     #             transform['sequence_length'] = new_num_frames
    #     #             break
    #     #     val_loop.dataloader_cfg = new_dataloader_cfg
    #     #     val_loop.dataloader = self.build_dataloader(train_loop.dataloader_cfg)