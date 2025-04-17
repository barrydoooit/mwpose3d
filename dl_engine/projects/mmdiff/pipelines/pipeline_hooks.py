from copy import deepcopy
from mmengine.registry import HOOKS

from mmengine.hooks import Hook
from mmengine.device import get_device

from typing import TYPE_CHECKING, Literal

import torch
import torch.backends.cudnn as cudnn

from dl_engine.projects.mmdiff.pipelines.ema import EMAHelper

if TYPE_CHECKING:
    from .train_loop import MMDiffTwoStageEpochBasedTrainLoop



@HOOKS.register_module()
class MMDiffPipelineHook(Hook):
    def before_train(self, runner):
        train_loop: 'MMDiffTwoStageEpochBasedTrainLoop' = runner.train_loop
        
        train_loop.grad_scaler = torch.GradScaler(device=get_device(), enabled=train_loop.phase1_cfg.get('amp', False))

        if train_loop.load_pretrain_from is not None and train_loop.pretrain_max_epochs == 0:
                checkpoint = torch.load(train_loop.load_pretrain_from, map_location=get_device(), weights_only=False)
                print("Loading pretrained feature encoder from: ", train_loop.load_pretrain_from)
                runner.model.load_state_dict(checkpoint, strict=True)
        
        train_loop.ema_helper = None if not train_loop.phase2_cfg.get('ema', False) else EMAHelper(
            mu=train_loop.phase2_cfg.get('ema_rate')
        )
        if train_loop.ema_helper is not None:
            train_loop.ema_helper.register(train_loop.runner.model)
    
    def after_train_epoch(self, runner):
        train_loop = runner.train_loop
        
        # Decay learning rate
        train_phase = 1 if train_loop._epoch < train_loop.pretrain_max_epochs else 2
        if train_phase == 2:
            decay_cfg = train_loop.phase2_cfg.get('lr_decay_cfg', None)
            if decay_cfg is not None:
                optimizer, step, lr, decay_step, gamma = \
                    runner.optimizer, train_loop.epoch, decay_cfg['lr'], decay_cfg['interval'], decay_cfg['gamma']
                lr = lr * gamma ** (step / decay_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
    
    def before_train_epoch(self, runner):
        train_loop = runner.train_loop

        if train_loop.epoch == train_loop.pretrain_max_epochs:
            new_num_frames = train_loop.phase2_cfg.get('num_frames')
            new_dataloader_cfg = deepcopy(train_loop.dataloader_cfg)
            for transform in new_dataloader_cfg['dataset']['pipeline']:
                if transform['type'] == 'SequenceClip':
                    transform['sequence_length'] = new_num_frames
                if transform['type'] == 'LoadMultiFrameFromH5':
                    transform['num_frames'] = new_num_frames
            new_dataloader_cfg['batch_size'] = train_loop.phase2_cfg.get('batch_size')
            new_dataloader_cfg['dataset']['sequence_length'] = new_num_frames + train_loop.phase2_cfg.get('backup_frames')
            train_loop.dataloader_cfg = new_dataloader_cfg
            train_loop.dataloader = runner.build_dataloader(train_loop.dataloader_cfg)

            new_lr = train_loop.phase2_cfg.get('lr')
            for param_group in runner.optimizer.param_groups:
                param_group['lr'] = new_lr
        if train_loop.epoch >= train_loop.pretrain_max_epochs:
            runner.model.model_feat.eval()
                
    def _before_epoch(self, runner, mode = 'train'):
        if mode == 'val':
            train_loop = runner._train_loop
            phase = 1 if train_loop.epoch <= train_loop.pretrain_max_epochs else 2 
            if phase == 1:
                runner.model.mode = 'pred_coarse'
            elif phase == 2:
                runner.model.mode = 'pred_fine'
            else:
                raise ValueError(f"Invalid phase: {phase}. Only 1 and 2 are supported.")
            return
        
        if mode == 'test':
            if "test_mode" in runner.cfg:
                test_mode: Literal['coarse', 'fine'] = runner.cfg.test_mode
                if test_mode == 'coarse':
                    runner.model.mode = 'pred_coarse'
                elif test_mode == 'fine':
                    runner.model.mode = 'pred_fine'
                else:
                    raise ValueError(f"Invalid test mode: {test_mode}. Only 'coarse' and 'fine' are supported.")