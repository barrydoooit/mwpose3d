import copy
from typing import Any, Callable, List, Sequence, Tuple, TYPE_CHECKING, Union, Deque, Optional
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
from mwpose3d.datasets.transforms.utils import apply_per_sample_transforms_serial
from mwpose3d.evaluation.postprocessing.base import BasePostProcessing, ComposePostProcess
from mwpose3d.registry import MODELS, TRANSFORMS, HOOKS
from mwpose3d.evaluation.postprocessing import POSTPROCESSING
import logging
from mwpose3d.utils.typing_utils import ConfigType

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
                transform = TRANSFORMS.build(dict(transform, online_mode=True))
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
                postprocess = POSTPROCESSING.build(dict(postprocess, online_mode=True))
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
    _current_engine: Optional['InferenceEngine'] = None

    def __init__(self,
                 model: dict,
                 preprocess_pipeline: List[dict],
                 load_from: str,
                 keypoints_involved: List[int],
                 frame_buffer_size: int,
                 postprocess_pipeline: Optional[List[dict]] = None,
                 cfg: Optional[ConfigType] = None,
                 custom_hooks: Optional[List[dict]] = None,
                 ): 
        InferenceEngine._current_engine = self
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        self.model: torch.nn.Module = MODELS.build(model)
        self.model.to(get_device())
        self.preprocess_pipeline: ComposePreprocessOnline = ComposePreprocessOnline(preprocess_pipeline) if preprocess_pipeline is not None else None
        self.postprocess_pipeline: Optional[ComposePostprocessOnline] = ComposePostprocessOnline(postprocess_pipeline) if postprocess_pipeline is not None else None
        if load_from is not None:
            self.load_checkpoint(load_from)
            self._loaded = True
        else:
            self._loaded = False
        self.model.eval()
        self.keypoints_involved = keypoints_involved
        self._custom_hooks: List[Hook] = []
        self.register_custom_hooks(custom_hooks)
        self.active_frames: Deque[Any] = deque(maxlen=frame_buffer_size)
        self._pred_history: Deque[Any] = deque(maxlen=frame_buffer_size)

        # NOTE: While your hooks is usually implemented for the runner to train/test offline, runner.xxx must also be available in this class.
        self.call_custom_hook('before_test')
        self.call_custom_hook('before_test_epoch')

    @property
    def loaded(self) -> bool:
        return self._loaded

    @classmethod
    def get_current_instance(cls) -> Optional['InferenceEngine']:
        return cls._current_engine

    def get_pred_history(self, n_recent: Optional[int] = None) -> Tuple[np.ndarray]:
        if n_recent is None:
            n_recent = len(self._pred_history) # All
        if n_recent <= 0 or len(self._pred_history) == 0:
            return tuple()
        n_recent = min(n_recent, len(self._pred_history))
        return tuple(list(self._pred_history)[-n_recent:])

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
            'remaining_frames_idx': list(range(len(self.active_frames))),
            }
        input = self.preprocess_pipeline(input_dict)
        input['pcd_frames'] = tuple([
            np.expand_dims(pcd_frame, axis=0) for pcd_frame in input['pcd_frames']
        ]) # make a batch dimension
        return input

    def infer(self, point_cloud: Union['SimplePointCloud5D', np.ndarray]) -> Optional[Union[Tuple[np.ndarray], np.ndarray]]:
        self.call_custom_hook('before_test_iter')
        if not self.loaded:
            logger.warning("InferenceEngine not loaded with weights yet. Call load_checkpoint() or pass the checkpoint file from 'load_from' first.")
        try:
            data_batch_dict = self._preprocess(point_cloud)
            if len(data_batch_dict) == 0:
                return None
            batch_inputs, data_samples = self.model.pack_input(data_batch_dict)
            with torch.no_grad():
                output = self.model(batch_inputs, data_samples, mode='predict')
            data_samples_0 = data_samples[0]
            batch_inputs, data_samples_0 = self._postprocess(batch_inputs, data_samples_0)
            preds = data_samples_0.pred.cpu().numpy()
            if preds.ndim == 2: # Sequence output
                preds = tuple(preds[i] for i in range(preds.shape[0]))
            if isinstance(preds, np.ndarray):
                self._pred_history.append(preds)
            elif isinstance(preds, tuple):
                for pred in preds:
                    self._pred_history.append(pred)
            return preds
        except RuntimeError as e:
            logger.warning(f"Inference Interupted: {e}")
            return None

    def infer_batched_dict(self, data_batch_dict: dict, inplace: bool = False) -> Optional[Union[Tuple[np.ndarray, ...], Tuple[Tuple[np.ndarray, ...], ...]]]:
        self.call_custom_hook('before_test_iter')
        if not self.loaded:
            logger.warning("InferenceEngine not loaded with weights yet. Call load_checkpoint() or pass the checkpoint file from 'load_from' first.")
        if not inplace:
            data_batch_dict = copy.deepcopy(data_batch_dict)
        if self.preprocess_pipeline is not None:
            data_batch_dict = apply_per_sample_transforms_serial(data_batch_dict, self.preprocess_pipeline.transforms)
        batch_inputs, data_samples = self.model.pack_input(data_batch_dict)
        with torch.no_grad():
            output = self.model(batch_inputs, data_samples, mode='predict')
        batch_inputs, data_samples = self._postprocess(batch_inputs, data_samples)
        preds = tuple([ds.pred.cpu().numpy() for ds in data_samples])
        if preds[0].ndim == 2:
            B, S = len(preds), preds[0].shape[0]
            preds = tuple(tuple([preds[b][s] for b in range(B)]) for s in range(S))
        else:
            preds = tuple([(pred,) for pred in preds])
        for pred in preds:
            for p in pred:
                self._pred_history.append(p)
        return preds


    def _postprocess(self, data_batch_dict: dict, datasample: 'SkeletonDataSample') -> Tuple[dict,  'SkeletonDataSample']:
        if self.postprocess_pipeline is None:
            return data_batch_dict, datasample
        data_batch_dict, datasample = self.postprocess_pipeline(data_batch_dict, datasample)
        return data_batch_dict, datasample

    def load_checkpoint(self, checkpoint: Union[str, torch.nn.Module, dict], strict: bool = True):
        if isinstance(checkpoint, str):
            state_dict = torch.load(checkpoint, map_location=torch.device(get_device()))
            self.model.load_state_dict(state_dict, strict=strict)
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint, strict=strict)
        elif isinstance(checkpoint, torch.nn.Module):
            self.model.load_state_dict(checkpoint.state_dict(), strict=strict)
        else:
            raise TypeError(f'checkpoint must be a str, dict or torch.nn.Module, but got {type(checkpoint)}')
        
    @classmethod
    def from_cfg(cls, config: dict):
        return cls(
            model=config['model'],
            frame_buffer_size=config['total_frames'],
            preprocess_pipeline=config['test_pipeline'],
            postprocess_pipeline=config.get('postprocess', None),
            load_from=config['load_from'],
            keypoints_involved=config['keypoints_involved'],
            custom_hooks=config.get('custom_hooks', None),
            cfg=config
        )