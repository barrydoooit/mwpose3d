
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
import os
from typing import Any, Callable, Dict, List, TYPE_CHECKING, Literal, Optional, Sequence, Tuple, Union
from mmengine.hooks import Hook
from mwpose3d.datasets.transforms.utils import  _apply_parallel_with_executor, _init_worker, apply_per_sample_transforms_serial
from mwpose3d.registry import HOOKS
import torch
import torch.nn as nn
from mmengine.dataset import Compose
from mwpose3d.runner.inference_engine import InferenceEngine
from mwpose3d.utils.typing_utils import ConfigType

if TYPE_CHECKING:
    from mwpose3d.runner import Runner



@HOOKS.register_module()
class ExtraTransformHook(Hook):
    def __init__(
        self,
        extra_pipeline,                      # Sequence[ConfigType | callable]
        *,
        parallel: bool = True,
        use_threads: bool = False,           # fallback if pickling is hard
        num_workers: Optional[int] = None,
        start_method: str = "spawn",
        share_cpu_tensors: bool = False,     # only for ProcessPool
        chunksize: Optional[int] = None,
        suppress_worker_warnings: bool = True,
    ):
        self.extra_pipeline = Compose(extra_pipeline)
        self.parallel = parallel
        self.use_threads = use_threads
        self.num_workers = num_workers or min(32, os.cpu_count() or 1)
        self.start_method = start_method
        self.share_cpu_tensors = share_cpu_tensors
        self.chunksize = chunksize
        self.suppress_worker_warnings = suppress_worker_warnings
        self._executor = None  # set in before_run

    def before_run(self, runner):
        if not self.parallel:
            return
        transforms = list(self.extra_pipeline.transforms)  # picklable list

        if self.use_threads:
            # threads share the same process; set global once here
            global _GLOBAL_TRANSFORMS
            _GLOBAL_TRANSFORMS = transforms
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        else:
            ctx = get_context(self.start_method)
            self._executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(transforms, self.suppress_worker_warnings),
            )

    def after_run(self, runner):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # Call this wherever your framework lets you postprocess the batch:
    def execute_extra_pipeline(self, data_batch: dict) -> dict:
        # SERIAL fallback
        if not self.parallel or self._executor is None:
            apply_per_sample_transforms_serial(
                data_batch,
                self.extra_pipeline.transforms,
                inplace=True,
            )

        # PARALLEL path (persistent pool; no repeated warnings)
        return _apply_parallel_with_executor(
            data_batch,
            executor=self._executor,
            inplace=True,
            chunksize=self.chunksize,
            share_cpu_tensors=self.share_cpu_tensors,
        )


