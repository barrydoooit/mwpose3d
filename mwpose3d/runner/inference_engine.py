import copy
from typing import Callable, List, Sequence, Tuple, TYPE_CHECKING, Union, Deque, Optional
from collections import deque
import numpy as np
import torch
from mmengine.device import get_device
from mmengine.dataset import Compose
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Priority, get_priority
from mwpose3d.datasets.skel_data_sample import SkeletonDataSample
from mwpose3d.datasets.transforms import BaseTransform, is_online_enabled
from mwpose3d.evaluation.postprocessing.base import BasePostProcessing, ComposePostProcess
from mwpose3d.registry import MODELS, TRANSFORMS, HOOKS
from mwpose3d.evaluation.postprocessing import POSTPROCESSING
import logging

from mwpose3d.runner.runner import ConfigType
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from mwpose3d.utils.pointcloud_toolkits import SimplePointCloud5D




class ComposePreprocessOnline(Compose):
    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            if isinstance(transform, dict):
                if not is_online_enabled(transform['type']): continue
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(f'transform should be a callable object, '
                                    f'but got {type(transform)}')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f'transform must be a callable object or dict, '
                    f'but got {type(transform)}')

class ComposePostprocessOnline(ComposePostProcess):
    def __init__(self, postprocesses: Optional[Sequence[Union[dict, Callable]]]):
        self.postprocesses: List[Callable] = []

        if postprocesses is None:
            postprocesses = []

        for postprocess in postprocesses:
            # `Compose` can be built with config dict with type and
            # corresponding arguments.
            if isinstance(postprocess, dict):
                if not is_online_enabled(postprocess['type']): continue
                postprocess = POSTPROCESSING.build(postprocess)
                if not callable(postprocess):
                    raise TypeError(f'postprocess should be a callable object, '
                                    f'but got {type(postprocess)}')
                self.postprocesses.append(postprocess)
            elif callable(postprocess):
                self.postprocesses.append(postprocess)
            else:
                raise TypeError(
                    f'postprocess must be a callable object or dict, '
                    f'but got {type(postprocess)}')
            
class InferenceEngine:
    def __init__(self,
                 model: dict,
                 preprocess_pipeline: List[dict],
                 load_from: str,
                 keypoints_involved: List[int],
                 post_process_pipeline: Optional[List[dict]] = None,
                 cfg: Optional[ConfigType] = None,
                 custom_hooks: Optional[List[dict]] = None,
                 frame_buffer_size: int = 10
                 ):    
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        self.model: torch.nn.Module = MODELS.build(model)
        self.model.to(get_device())
        self.preprocess_pipeline: list['BaseTransform'] = ComposePreprocessOnline(preprocess_pipeline)
        self.post_process_pipeline: list['BasePostProcessing'] = ComposePostprocessOnline(post_process_pipeline) if post_process_pipeline is not None else []
        self.model.load_state_dict(torch.load(load_from, map_location=torch.device(get_device())))
        self.model.eval()
        self.keypoints_involved = keypoints_involved
        self._custom_hooks: List[Hook] = []
        self.register_custom_hooks(custom_hooks)

        self.active_frames: Deque[np.ndarray] = deque(maxlen=frame_buffer_size)

        # NOTE: While your hooks is usually implemented for the runner to train/test offline, runner.xxx must also be available in this class.
        self.call_custom_hook('before_test')
        self.call_custom_hook('before_test_epoch')
        self.call_custom_hook('before_test_model')

    
    def add_frame(self, point_cloud: np.ndarray):
        self.active_frames.append(point_cloud)

    @property
    def custom_hooks(self) -> List[Hook]:
        return self._custom_hooks
    
    def register_custom_hooks(self, custom_hooks: Optional[List[dict]] = None):
        if custom_hooks is None:
            return
        for hook in custom_hooks:
            _priority = None
            if isinstance(hook, dict):
                if 'priority' in hook:
                    _priority = hook.pop('priority')

                hook_obj = HOOKS.build(hook)
            else:
                hook_obj = hook
            if _priority is not None:
                hook_obj.priority = _priority
            
            inserted = False
            for i in range(len(self._custom_hooks) - 1, -1, -1):
                if get_priority(hook_obj.priority) >= get_priority(self._custom_hooks[i].priority):
                    self._custom_hooks.insert(i + 1, hook_obj)
                    inserted = True
                    break
            if not inserted:
                self._custom_hooks.insert(0, hook_obj)
    
    def call_custom_hook(self, fn_name: str, **kwargs) -> None:
        for hook in self.custom_hooks:
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None
                
    def _preprocess(self, point_cloud: Union['SimplePointCloud5D', np.ndarray]) -> dict:
        point_cloud = point_cloud if isinstance(point_cloud, np.ndarray) \
            else np.asarray(point_cloud.serialize(compact=True))
        self.add_frame(point_cloud)
        if len(self.active_frames) < self.active_frames.maxlen:
            return dict()
        input_dict = {
            'pcd_frames': tuple(self.active_frames),
            }
        input = self.preprocess_pipeline(input_dict)
        input['pcd_frames'] = tuple([
            np.expand_dims(pcd_frame, axis=0) for pcd_frame in input['pcd_frames']
        ]) # make a batch dimension
        return input
    
    def infer(self, point_cloud: Union['SimplePointCloud5D', np.ndarray]) -> Optional[np.ndarray]:
        try:
            data_batch_dict = self._preprocess(point_cloud)
            if len(data_batch_dict) == 0:
                return None
            
            batch_inputs, data_samples = self.model.pack_input(data_batch_dict)
            with torch.no_grad():
                output = self.model(batch_inputs, data_samples, mode='predict')
            data_samples_0 = data_samples[0]
            data_batch_dict, data_samples_0 = self._postprocess(data_batch_dict, data_samples_0)
            return data_samples_0.pred.cpu().numpy() if len(data_samples) > 0 else None
        except RuntimeError as e:
            logger.warning(f"Inference Interupted: {e}")
            return None

    def _postprocess(self, data_batch_dict: dict, datasample: 'SkeletonDataSample') -> Tuple[dict,  'SkeletonDataSample']:
        if len(self.post_process_pipeline) == 0:
            return data_batch_dict, datasample
        data_batch_dict, datasample = self.post_process_pipeline(data_batch_dict, datasample)
        return data_batch_dict, datasample
        
        return data_batch_dict, datasample
    @classmethod
    def from_cfg(cls, config: dict):
        return cls(
            model=config['model'],
            preprocess_pipeline=config['test_pipeline'],
            load_from=config['load_from'],
            keypoints_involved=config['keypoints_involved'],
            custom_hooks=config.get('custom_hooks', None),
            cfg=config
        )