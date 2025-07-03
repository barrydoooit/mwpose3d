from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union, TYPE_CHECKING

from mwpose3d.registry import LOOPS
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from mwpose3d.runner.runner import Runner


@LOOPS.register_module()
class BaseLoop(metaclass=ABCMeta):
    def __init__(self, runner: 'Runner', dataloader: Union[DataLoader, Dict], out_file: str | None = None) -> None:
        self._runner = runner
        if isinstance(dataloader, dict):
            self.dataloader = runner.build_dataloader(
                dataloader)
        else:
            self.dataloader = dataloader
        self._epoch = 0
        self.out_file = out_file
        if self.out_file is not None:
            self.out_file = Path(self.runner.work_dir / self.out_file)
            self.out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_file, 'w') as f:
                f.write('epoch, train_loss, validation_loss\n')

    @property
    def runner(self):
        return self._runner

    @abstractmethod
    def run(self) -> Any:
        """Execute loop."""

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    def _write_training_progress_to_file(self, epoch: int, train_loss: float, validation_loss: float):
        if validation_loss is None or self.out_file is None:
            return

        with open(self.out_file, 'a') as f:
            f.write(f'{epoch}, {train_loss}, {validation_loss}\n')

    def validate(self, filename: str | None = None, mode: str = "loss") -> float | None:
        loss: float | None = self.runner.val_loop.run(mode=mode)

        if filename:
            self.runner.save_checkpoint(filename)
        else:
            self.runner.save_checkpoint(f'epoch_{self._epoch}.pth')
        return loss