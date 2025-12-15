from typing import TYPE_CHECKING, Dict, Union
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mwpose3d.registry import LOOPS
from .base_loop import BaseLoop

if TYPE_CHECKING:
    from mwpose3d.runner.runner import Runner


@LOOPS.register_module()
class FastEvalLoop(BaseLoop):
    def __init__(
            self,
            runner: 'Runner',
            dataloader: Union[DataLoader, Dict]
    ):
        super().__init__(runner, dataloader)
        self._reset()

    def _reset(self):
        self._loss = 0
        self._sum_loss = 0
        self._loss_count = 0

    def run(self, mode: str = 'loss') -> float:
        """
        Evaluate the model and return the loss
        """
        self._reset()
        self._run_epoch(mode)
        return self._loss

    def _run_epoch(self, mode: str) -> None:
        self.runner.model.eval()
        with torch.no_grad():
            for idx, data_batch in tqdm(enumerate(self.dataloader),
                          total=len(self.dataloader),
                          desc='Validating'):
                self._run_iter(idx, data_batch, mode)

    def _run_iter(self, idx: int, data_batch: dict, mode: str) -> None:
        batch_inputs, data_samples = self.runner.model.pack_input(data_batch)
        loss = self.runner.model(batch_inputs, data_samples, mode=mode)  # Breaks with ptrans
        self._sum_loss += loss.item()
        self._loss_count += 1
        self._loss = self._sum_loss / self._loss_count
