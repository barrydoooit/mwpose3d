
from typing import Any, Callable, Dict, List, TYPE_CHECKING, Literal, Optional, Sequence, Tuple, Union
from mmengine.hooks import Hook
from mwpose3d.datasets.transforms.utils import apply_per_sample_transforms_serial
from mwpose3d.registry import HOOKS
import torch
import torch.nn as nn
from mmengine.dataset import Compose
from .extra_transform_hook import ExtraTransformHook
from mwpose3d.runner.inference_engine import InferenceEngine
from mwpose3d.utils.typing_utils import ConfigType

if TYPE_CHECKING:
    from mwpose3d.runner import Runner


@HOOKS.register_module()
class PreInferenceHook(ExtraTransformHook):
    PREINFERENCE_RESULTS = 'preinference_results'
    def __init__(self,
                 inference_engine: Union[InferenceEngine, ConfigType],
                 extra_pipeline: Sequence[Union[ConfigType, callable]],
                 earliest_activation_epoch: int = 0,
                 dynamic_loading_start_epoch: int = -1,
                 strict_loading: bool = True,
                 **kwargs
                 ):
        super().__init__(extra_pipeline, **kwargs)
        self.earliest_activation_epoch = earliest_activation_epoch
        if isinstance(inference_engine, InferenceEngine):
            self.inference_engine = inference_engine
        else:
            self.inference_engine = InferenceEngine.from_cfg(inference_engine)
        self.dynamic_loading_start_epoch = dynamic_loading_start_epoch if dynamic_loading_start_epoch >= 0 else float('inf')
        self.strict_loading = strict_loading
        self.activated = False
    
    def load_checkpoint(self, checkpoint: Union[str, torch.nn.Module, dict], strict: bool):
        self.inference_engine.load_checkpoint(checkpoint, strict=strict)
    
    def load_checkpoint_from_runner(self, runner: 'Runner', strict: bool = True):
        self.load_checkpoint(runner.model.state_dict(), strict=strict)

    def before_train_epoch(self, runner):
        if runner.epoch == 0 and not self.inference_engine.loaded:
            return
        
        if runner.epoch >= self.earliest_activation_epoch:
            self.activated = True
        
        if runner.epoch >= self.dynamic_loading_start_epoch and self.activated:
            self.load_checkpoint_from_runner(runner, strict=self.strict_loading)
    
    def before_val_epoch(self, runner):
        self.load_checkpoint_from_runner(runner, strict=self.strict_loading)
        self.activated = True
    
    def before_test_epoch(self, runner):
        self.load_checkpoint_from_runner(runner, strict=self.strict_loading)
        self.activated = True
    
    def _before_iter(self, runner, batch_idx, data_batch, mode = 'train'):
        if mode == 'train' and not self.activated:
            return
        batched_preds = self.inference_engine.infer_batched_dict(data_batch, inplace=False)
        data_batch[self.PREINFERENCE_RESULTS] = batched_preds
        super()._before_iter(runner, batch_idx, data_batch, mode=mode)


