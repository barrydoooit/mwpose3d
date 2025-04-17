from typing import Protocol

from mwpose3d.datasets.skel_data_sample import SkeletonDataSample


class BaseMetric(Protocol):
    def process_sample(self, data_sample: SkeletonDataSample, data_batch: dict = None) -> None:
        ...
    def evaluate(self) -> dict:
        ...
    def reset(self) -> None:
        ...