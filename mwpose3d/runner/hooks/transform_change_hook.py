import json
from pathlib import Path
import time
from typing import Any, Dict, List, TYPE_CHECKING, Literal, Optional, Sequence, Tuple
from mmengine.hooks import Hook
from mwpose3d.registry import HOOKS
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from mwpose3d.runner import Runner


@HOOKS.register_module()
class TransformChangeHookPerTrainEpoch(Hook):
    def __init__(self, 
                 method_spec: str,
                 method_kwargs: dict = None,
                 starting_epoch: int = 0,):
        super().__init__()
        self.method_spec = method_spec
        self.method_kwargs = method_kwargs or {}
        self.starting_epoch = starting_epoch
        self._resolved: Tuple[object, callable] = None

    def _parse_spec(self):
        try:
            class_name, method_name = self.method_spec.split(".", 1)
        except ValueError as e:
            raise ValueError(
                f"method_spec must be 'ClassName.method_name', got: {self.method_spec}"
            ) from e
        return class_name, method_name

    def _resolve(self, runner):
        loop = runner.train_loop
        dataloader = loop.dataloader
        transforms = dataloader.dataset.pipeline.transforms

        class_name, method_name = self._parse_spec()
        for t in transforms:
            if t.__class__.__name__ == class_name:
                method = getattr(t, method_name, None)
                if callable(method):
                    self._resolved = (t, method)
                    return

    def before_train(self, runner) -> None:
        self._resolve(runner)

    def _after_epoch(self, runner, mode: str) -> None:
        if self._resolved is None:
            return
        epoch = runner.epoch
        if epoch < self.starting_epoch:
            return
        transform_instance, bound_method = self._resolved
        kwargs = dict(self.method_kwargs)  # copy to be safe
        try:
            bound_method(runner=runner, **kwargs)
        except TypeError:
            bound_method(**kwargs)