from typing import TYPE_CHECKING, Dict, Sequence, Union
import torch
from torch.utils.data import DataLoader
from .base_loop import BaseLoop, LOOPS

if TYPE_CHECKING:
    from dl_engine.runner.runner import Runner

@LOOPS.register_module()
class EpochBasedTrainLoop(BaseLoop):
    def __init__(
        self,
        runner: 'Runner',
        dataloader: Union[DataLoader, Dict],
        max_epochs: int,
        val_begin: int = 1,
        val_interval: int = 1
    ):
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        self.stop_training = False
        
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
        while self._epoch < self._max_epochs and not self.stop_training:
            self._run_epoch()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and (self._epoch % self.val_interval == 0
                         or self._epoch == self._max_epochs)):
                self.runner.val_loop.run()
        return self.runner.model
    
    def _run_epoch(self) -> None:
        self.runner.model.train()
        if self._epoch % 50 == 0:
            print(f'Epoch [{self._epoch}/{self._max_epochs}]')
        for idx, data_batch in enumerate(self.dataloader):
            self._run_iter(idx, data_batch)
        self._epoch += 1
    
    def _run_iter(self, idx: int, data_batch: dict) -> None:
        assert hasattr(self.runner.model, 'pack_input')
        batch_inputs, data_samples = self.runner.model.pack_input(data_batch)
        loss = self.runner.model(batch_inputs, data_samples, mode='loss')
        if self._iter % 1000 == 0:
            print(f'Iter [{self._iter}/{self._max_iters}] Loss: {loss.item()}')
        self.runner.optimizer.zero_grad()
        loss.backward()
        self.runner.optimizer.step()
        self._iter += 1
        
        
        