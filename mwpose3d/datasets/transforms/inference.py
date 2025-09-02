from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch

from mwpose3d.datasets.transforms.loading import LoadMultiFrameFromH5
from mwpose3d.registry import TRANSFORMS
from mwpose3d.datasets.transforms import BaseTransform
from mwpose3d.datasets.transforms.base import OnlineEnabled
from mwpose3d.runner.inference_engine import InferenceEngine

if TYPE_CHECKING:
    from mwpose3d.runner.runner import Runner


@OnlineEnabled
@TRANSFORMS.register_module()
class Inference(BaseTransform):

    PREDICTIONS_KEY: str = "inference_predictions"

    def __init__(self,
                 inference_engine: Optional[Union[InferenceEngine, Dict[str, Any]]] = None,
                 online_mode: bool = False,):
        super().__init__(online_mode)

        self._engine: Optional[InferenceEngine] = inference_engine
        if self.online_mode:
            self._engine = InferenceEngine.get_current_instance()
        else:
            if not isinstance(inference_engine, InferenceEngine):
                if inference_engine.get("load_from", None) is not None:
                    self._engine = InferenceEngine.from_cfg(inference_engine)
                else:
                    self._engine = InferenceEngine.from_cfg(dict(inference_engine, load_from=None))

    @property
    def engine(self) -> Optional[InferenceEngine]:
        if self._engine.loaded:
            return self._engine
        return None

    def transform(self, input: dict) -> Dict[str, Any]:
        if self.engine is None:
            return input
        pcd_frames = input[LoadMultiFrameFromH5.POINT_CLOUDS_KEY]
        if not isinstance(pcd_frames, (list, tuple)):
            raise TypeError(f"Expected list or tuple for {LoadMultiFrameFromH5.POINT_CLOUDS_KEY}, "
                            f"but got {type(pcd_frames)}")

        input[self.PREDICTIONS_KEY] = self._engine.infer_new_sequence(pcd_frames)
        return input

    def transform_online(self, input: dict) -> Dict[str, Any]:
        preds = self._engine.get_pred_history()
        input[self.PREDICTIONS_KEY] = preds
        return input

    def load_checkpoint(self, checkpoint: Union[str, torch.nn.Module, dict], strict: bool):
        self._engine.load_checkpoint(checkpoint, strict=strict)
    
    def load_checkpoint_from_runner(self, runner: Runner, strict: bool = True):
        self.load_checkpoint(runner.model.state_dict(), strict=strict)