from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from mmengine.device import get_device
from .ema import EMAHelper
from dl_engine.runner.base_loop import BaseLoop, LOOPS
from . import utils



if TYPE_CHECKING:
    from dl_engine.runner.runner import Runner

@LOOPS.register_module()
class MMDiffTwoStageEpochBasedTrainLoop(BaseLoop):
    def __init__(
        self,
        runner: 'Runner',
        dataloader: dict,
        pretrain_max_epochs: int,
        train_max_epochs: int,
        val_interval: int,
        phase_cfg: dict = {},
        load_pretrain_from: Optional[str] = None,
        val_begin: int = 1,
    ):
        assert isinstance(dataloader, dict), f"For {self.__class__.__name__}, `dataloader` should be a dict, but got {type(dataloader)}."
        self.dataloader_cfg = deepcopy(dataloader)
        super().__init__(runner, dataloader)
        self.pretrain_max_epochs = pretrain_max_epochs
        self.train_max_epochs = train_max_epochs
        self._max_epochs = int(pretrain_max_epochs + train_max_epochs)
        assert self._max_epochs == pretrain_max_epochs + train_max_epochs, \
            f'`max_epochs` should be a integer number, but get {pretrain_max_epochs + train_max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_interval = val_interval
        self.val_begin = val_begin
        self.load_pretrain_from = load_pretrain_from

        self.phase_cfg = deepcopy(phase_cfg)
        self.phase1_cfg = self.phase_cfg.get('phase1', {})
        self.phase2_cfg = self.phase_cfg.get('phase2', {})

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        self.runner.call_hook('before_train')
        epoch_pbar = tqdm(total=self._max_epochs, desc='Training Progress', unit='epoch', position=0)
        
        while self._epoch < self._max_epochs:
            self._run_epoch(epoch_pbar)
            if self._epoch <= self.pretrain_max_epochs:
                current_phase = 1
                phase_epochs = self.pretrain_max_epochs
                val_interval = self.val_interval
            else:
                current_phase = 2
                phase_epochs = self.train_max_epochs
                val_interval = self.val_interval

            phase_epoch_idx = self._epoch if current_phase == 1 else self._epoch - self.pretrain_max_epochs
            if (self.runner.val_loop is not None
                    and phase_epoch_idx >= self.val_begin
                    and (phase_epoch_idx % val_interval == 0
                        or phase_epoch_idx == phase_epochs)):
                self.runner.val_loop.run()
                self.runner.save_checkpoint(f'phase_{current_phase}-epoch_{self._epoch - (current_phase - 1) * self.pretrain_max_epochs}.pth')
        
        epoch_pbar.close()
        self.runner.call_hook('after_train')
        return self.runner.model

    def _run_epoch(self, epoch_pbar) -> None:
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        current_phase = 1 if self._epoch < self.pretrain_max_epochs else 2
        run_iter = self._run_phase_1_iter if current_phase == 1 else self._run_phase_2_iter
        
        sliding_window = deque(maxlen=10)  # Store last 10 loss values
        
        for idx, data_batch in enumerate(self.dataloader):
            loss = run_iter(idx, data_batch)
            sliding_window.append(loss.item())
            avg_loss = sum(sliding_window) / len(sliding_window)
            epoch_pbar.set_postfix(avg_loss=f'{avg_loss:.4f}')
        
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1
        epoch_pbar.update(1)

    def _run_phase_1_iter(self, idx: int, data_batch: dict) -> torch.Tensor:
        self.runner.call_hook('before_train_iter', batch_idx=idx, data_batch=data_batch)
        assert hasattr(self.runner.model, 'pack_input')
        batch_inputs, data_samples = self.runner.model.pack_input(data_batch)
        loss = self.runner.model(batch_inputs, data_samples, mode='loss-pretrain')
        
        self.runner.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.runner.model.parameters(), self.phase1_cfg.get('grad_clip', 1.0))
        self.grad_scaler.step(self.runner.optimizer)
        self.grad_scaler.update()
        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=loss,)
        self._iter += 1
        return loss  # Return loss for tracking

    def _run_phase_2_iter(self, idx: int, data_batch: dict) -> torch.Tensor:
        assert hasattr(self.runner.model, 'pack_input')
        batch_inputs, data_samples = self.runner.model.pack_input(data_batch)
        loss = self.runner.model(batch_inputs, data_samples, mode='loss-train')
        
        self.runner.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.runner.model.parameters(), self.phase2_cfg.get('grad_clip', 1.0))
        self.runner.optimizer.step()
        
        if self.ema_helper is not None:
            self.ema_helper.update(self.runner.model)
        
        self._iter += 1
        return loss  # Return loss for tracking
        
    # hooks. To be transferred after TODO: implement hook manager in runner

    # def hook_before_train(self) -> None:
    #     self.grad_scaler = torch.GradScaler(device=get_device(), enabled=self.phase1_cfg.get('amp', False))

    #     if self.load_pretrain_from is not None:
    #         raise NotImplementedError("TODO: load pretrain from checkpoint and skip phase 1")
        
    #     self.ema_helper = None if not self.phase2_cfg.get('ema', False) else EMAHelper(
    #         mu=self.phase2_cfg.get('ema_rate')
    #     )
    #     if self.ema_helper is not None:
    #         self.ema_helper.register(self.runner.model)
            
    # def hook_after_train_epoch(self) -> None:
    #     self = self.runner
    #     train_loop = self.train_loop

    #     # Decay learning rate
    #     train_phase = 1 if train_loop._epoch < train_loop.pretrain_max_epochs else 2
    #     if train_phase == 2:
    #         decay_cfg = train_loop.phase2_cfg.get('lr_decay_cfg', None)
    #         if decay_cfg is not None:
    #             optimizer, step, lr, decay_step, gamma = \
    #                 self.optimizer, train_loop.epoch, decay_cfg['lr'], decay_cfg['interval'], decay_cfg['gamma']
    #             lr = lr * gamma ** (step / decay_step)
    #             for param_group in optimizer.param_groups:
    #                 param_group['lr'] = lr

    # def hook_before_train_epoch(self) -> None:
    #     self = self.runner
    #     train_loop = self.train_loop

    #     if train_loop.epoch == train_loop.pretrain_max_epochs:
    #         new_num_frames = train_loop.phase2_cfg.get('num_frames')
    #         new_dataloader_cfg = deepcopy(train_loop.dataloader_cfg)
    #         for transform in new_dataloader_cfg['dataset']['pipeline']:
    #             if transform['type'] == 'SequenceClip':
    #                 transform['sequence_length'] = new_num_frames
    #             if transform['type'] == 'LoadMultiFrameFromH5':
    #                 transform['num_frames'] = new_num_frames
    #         new_dataloader_cfg['batch_size'] = train_loop.phase2_cfg.get('batch_size')
    #         new_dataloader_cfg['dataset']['sequence_length'] = new_num_frames + train_loop.phase2_cfg.get('backup_frames')
    #         train_loop.dataloader_cfg = new_dataloader_cfg
    #         train_loop.dataloader = self.build_dataloader(train_loop.dataloader_cfg)

    #         new_lr = train_loop.phase2_cfg.get('lr')
    #         for param_group in self.optimizer.param_groups:
    #             param_group['lr'] = new_lr
