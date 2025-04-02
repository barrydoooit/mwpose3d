from typing import Protocol
from mmengine.registry import Registry

from dl_engine.dataset.skel_data_sample import SkeletonDataSample

METRICS = Registry('metrics')

class BaseMetric(Protocol):
    def process_sample(self, data_sample: SkeletonDataSample, data_batch: dict = None) -> None:
        ...
    def evaluate(self) -> dict:
        ...
    def reset(self) -> None:
        ...