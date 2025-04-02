import copy
from logging import config
from pathlib import Path
from typing import Dict, List, Optional, Union
from mmengine.config import Config, ConfigDict
from mmengine.device import get_device
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Priority, get_priority
import torch
from torch.utils.data import DataLoader
from dl_engine.dataset.base import DATASETS
from dl_engine.dataset.utils import pseudo_collate
from dl_engine.model.base import MODELS
from dl_engine.runner.base_loop import LOOPS, BaseLoop
from dl_engine.runner.train_loop import EpochBasedTrainLoop



ConfigType = Union[Dict, Config, ConfigDict]

class Runner:
    def __init__(self,
                 model: dict,
                 work_dir: str,
                 train_dataloader: Optional[dict] = None,
                 val_dataloader: Optional[dict] = None,
                 test_dataloader: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 val_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
                 custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
                 optimizer_cfg: Optional[dict] = None,
                 load_from: Optional[str] = None,
                 cfg: Optional[ConfigType] = None,):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.model: torch.nn.Module = MODELS.build(model)
        self.model.to(get_device())
        
        self._train_dataloader = train_dataloader
        self._train_loop = train_cfg
        self._optimizer_cfg = optimizer_cfg
        
        self._val_dataloader = val_dataloader
        self._val_loop = val_cfg
        
        self._test_dataloader = test_dataloader
        self._test_loop = test_cfg
        self._load_from = load_from
        
        self._hooks: List[Hook] = []
        self.register_hooks(default_hooks, custom_hooks)

        self.cfg = copy.deepcopy(cfg) if cfg is not None else None
        
    @staticmethod
    def build_dataloader(dataloader_cfg: dict):
        dataset_cfg = dataloader_cfg.pop('dataset')
        assert dataset_cfg.get('type') == 'MotionDataset'
        dataset = DATASETS.build(dataset_cfg)
        
        dataloader = DataLoader(
            dataset,
            batch_size=dataloader_cfg['batch_size'],
            shuffle=dataloader_cfg['shuffle'],
            collate_fn=pseudo_collate,
            num_workers=dataloader_cfg['num_workers'],
        )
        return dataloader
    
    def build_train_loop(self, loop_cfg: dict):
        loop_cfg = copy.deepcopy(loop_cfg)
        if 'type' in loop_cfg:
            loop = LOOPS.build(loop_cfg,
                               default_args=dict(runner=self, dataloader=self._train_dataloader))
        else:
            by_epoch = loop_cfg.pop('by_epoch', False)
            if by_epoch:
                loop = EpochBasedTrainLoop(**loop_cfg, runner=self, dataloader=self._train_dataloader)
            else:
                raise NotImplementedError
            
        self._optimizer_cfg['params'] = self.model.parameters()
        optim_cls = getattr(torch.optim, self._optimizer_cfg.pop('type'))
        self.optimizer = optim_cls(**self._optimizer_cfg)
        return loop
    
    @property
    def train_loop(self):
        """:obj:`BaseLoop`: A loop to run training."""
        if isinstance(self._train_loop, BaseLoop) or self._train_loop is None:
            return self._train_loop
        else:
            self._train_loop = self.build_train_loop(self._train_loop)
            return self._train_loop

    @property
    def val_loop(self):
        """:obj:`BaseLoop`: A loop to run validation."""
        if isinstance(self._val_loop, BaseLoop) or self._val_loop is None:
            return self._val_loop
        else:
            self._val_loop = LOOPS.build(self._val_loop,
                               default_args=dict(runner=self, dataloader=self._val_dataloader))
            return self._val_loop

    @property
    def test_loop(self):
        """:obj:`BaseLoop`: A loop to run testing."""
        if isinstance(self._test_loop, BaseLoop) or self._test_loop is None:
            return self._test_loop
        else:
            self._test_loop = LOOPS.build(self._test_loop,
                               default_args=dict(runner=self, dataloader=self._test_dataloader))
            return self._test_loop
    
    @classmethod
    def from_cfg(cls, cfg: ConfigType):
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            default_hooks=cfg.get('default_hooks', None),
            custom_hooks=cfg.get('custom_hooks', None),
            optimizer_cfg=cfg.get('optimizer_cfg'),
            load_from=cfg.get('load_from', None),
            cfg=cfg
        )
        return runner
    
    def train(self):
        print('Start training')
        self.train_loop.run()
        print('Training finished')
        

    def test(self):
        print('Start testing')
        self.load_checkpoint(self._load_from)
        self.test_loop.run()
        print('Testing finished')

    def save_checkpoint(self, filename: str):
        torch.save(self.model.state_dict(), self.work_dir / filename)
    
    def load_checkpoint(self, filename: str):
        self.model.load_state_dict(torch.load(filename, 
                                              map_location=torch.device(get_device()),
                                              weights_only=True))
    
    @property
    def hooks(self) -> List[Hook]:
        return self._hooks
    
    def register_hook(
            self,
            hook: Union[Hook, Dict],
            priority: Optional[Union[str, int, Priority]] = None) -> None:

        if not isinstance(hook, (Hook, dict)):
            raise TypeError(
                f'hook should be an instance of Hook or dict, but got {hook}')

        _priority = None
        if isinstance(hook, dict):
            if 'priority' in hook:
                _priority = hook.pop('priority')

            hook_obj = HOOKS.build(hook)
        else:
            hook_obj = hook

        if priority is not None:
            hook_obj.priority = priority
        elif _priority is not None:
            hook_obj.priority = _priority

        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if get_priority(hook_obj.priority) >= get_priority(
                    self._hooks[i].priority):
                self._hooks.insert(i + 1, hook_obj)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook_obj)

    def register_default_hooks(
            self,
            hooks: Optional[Dict[str, Union[Hook, Dict]]] = None) -> None:

        default_hooks: dict = dict(
            # runtime_info=dict(type='RuntimeInfoHook'),
            # timer=dict(type='IterTimerHook'),
            # sampler_seed=dict(type='DistSamplerSeedHook'),
            # logger=dict(type='LoggerHook'),
            # param_scheduler=dict(type='ParamSchedulerHook'),
            # checkpoint=dict(type='CheckpointHook', interval=1),
        )
        if hooks is not None:
            for name, hook in hooks.items():
                if name in default_hooks and hook is None:
                    # remove hook from _default_hooks
                    default_hooks.pop(name)
                else:
                    assert hook is not None
                    default_hooks[name] = hook

        for hook in default_hooks.values():
            self.register_hook(hook)

    def register_custom_hooks(self, hooks: List[Union[Hook, Dict]]) -> None:
        for hook in hooks:
            self.register_hook(hook)

    def register_hooks(
            self,
            default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
            custom_hooks: Optional[List[Union[Hook, Dict]]] = None) -> None:
        self.register_default_hooks(default_hooks)

        if custom_hooks is not None:
            self.register_custom_hooks(custom_hooks)
    
    def call_hook(self, fn_name: str, **kwargs) -> None:
        for hook in self._hooks:
            # support adding additional custom hook methods
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None