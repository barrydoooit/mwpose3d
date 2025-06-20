from typing import TYPE_CHECKING, Dict, Sequence, Union, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base_loop import BaseLoop
from mwpose3d.registry import LOOPS
from pathlib import Path
from mwpose3d.runner.fast_eval_loop import FastEvalLoop

if TYPE_CHECKING:
    from mwpose3d.runner.runner import Runner

@LOOPS.register_module()
class EpochBasedTrainLoop(BaseLoop):
    def __init__(
        self,
        runner: 'Runner',
        dataloader: Union[DataLoader, Dict],
        max_epochs: int,
        val_begin: int = 1,
        val_interval: int = 1,
        out_file: Optional[str] = None
    ):
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._epoch_loss = 0
        self._epoch_sum_loss = 0
        self._epoch_loss_count = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        self.stop_training = False
        self.out_file = out_file
        if self.out_file is not None:
            self.out_file = Path(self.runner.work_dir / self.out_file)
            if not self.out_file.parent.exists():
                self.out_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                with open(self.out_file, 'w') as f:
                    f.write('epoch, train_loss, validation_loss\n')


        
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

    def _update_out_file(self, train_loss, validation_loss):
        with open(self.out_file, 'a') as f:
            f.write(f'{self.epoch}, {train_loss}, {validation_loss}\n')

    def run(self) -> torch.nn.Module:
        self.runner.call_hook('before_train')
        
        self.epoch_pbar = tqdm(
        range(1, self._max_epochs + 1),
        desc='Epochs',
        leave=True,
        total=self._max_epochs
        )
    
        for epoch in self.epoch_pbar:
            if self.stop_training:
                break
            self._run_epoch()
            #self.epoch_pbar.update(1)
            
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and (self._epoch % self.val_interval == 0
                         or self._epoch == self._max_epochs)):
                ret = self.runner.val_loop.run()
                if isinstance(self.runner.val_loop, FastEvalLoop) and self.out_file is not None:
                    self._update_out_file(self._epoch_loss, ret)
                self.runner.save_checkpoint(f'epoch_{self._epoch}.pth')
        
        self.epoch_pbar.close()
        self.runner.call_hook('after_train')
        return self.runner.model


    def _run_epoch(self) -> None:
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        self._epoch_sum_loss = 0
        self._epoch_loss_count = 0
        for idx, data_batch in enumerate(self.dataloader):
            self._run_iter(idx, data_batch)
            
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1
    
    def _run_iter(self, idx: int, data_batch: dict) -> None:
        self.runner.call_hook('before_train_iter', batch_idx=idx, data_batch=data_batch)
        assert hasattr(self.runner.model, 'pack_input')
        batch_inputs, data_samples = self.runner.model.pack_input(data_batch)
        loss = self.runner.model(batch_inputs, data_samples, mode='loss')
        self.runner.optimizer.zero_grad()
        loss.backward()
        self.runner.optimizer.step()
        self._epoch_sum_loss += loss.item()
        self._epoch_loss_count += 1
        self._epoch_loss = self._epoch_sum_loss / self._epoch_loss_count
        if self._iter % 30 == 0:
            self.epoch_pbar.set_postfix_str(f'loss: {self._epoch_loss:.4f}')
        )
        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=loss
        self._iter += 1
        
        
        
