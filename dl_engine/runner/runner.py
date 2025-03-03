import copy
from logging import config
from pathlib import Path
from typing import Dict, Optional, Union
from mmengine.config import Config, ConfigDict
from mmengine.device import get_device
import torch
from torch.utils.data import DataLoader
from dl_engine.dataset.base import DATASETS
from dl_engine.dataset.utils import pseudo_collate
from dl_engine.models.base import MODELS
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
            optimizer_cfg=cfg.get('optimizer_cfg'),
            load_from=cfg.get('load_from', None),
            cfg=cfg
        )
        return runner
    
    def train(self):
        print('Start training')
        self.train_loop.run()
        print('Training finished')
        self.save_checkpoint('latest.pth')

    def test(self):
        print('Start testing')
        self.load_checkpoint(self._load_from)
        self.test_loop.run()
        print('Testing finished')

    def save_checkpoint(self, filename: str):
        torch.save(self.model.state_dict(), self.work_dir / filename)
    
    def load_checkpoint(self, filename: str):
        self.model.load_state_dict(torch.load(filename))