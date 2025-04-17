from mmengine.registry import Registry

from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS


LOOPS = Registry(
    # TODO: update the location when mwpose3d has its own loop
    'loop',
    parent=MMENGINE_LOOPS)
HOOKS = Registry(
    'hook', parent=MMENGINE_HOOKS, locations=['mmdet3d.engine.hooks'])
DATASETS = Registry(
    'dataset', parent=MMENGINE_DATASETS, locations=['mwpose3d.datasets'])
TRANSFORMS = Registry(
    'transform', parent=MMENGINE_TRANSFORMS, locations=['mwpose3d.datasets'])
MODELS = Registry(
    'model', parent=MMENGINE_MODELS, locations=['mwpose3d.models'])
METRICS = Registry(
    'metric', parent=MMENGINE_METRICS, locations=['mwpose3d.evaluation'])