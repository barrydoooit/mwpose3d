from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Union, TYPE_CHECKING

from mwpose3d.registry import LOOPS
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from mwpose3d.runner.runner import Runner


@LOOPS.register_module()
class BaseLoop(metaclass=ABCMeta):
    def __init__(self, runner: 'Runner', dataloader: Union[DataLoader, Dict]) -> None:
        self._runner = runner
        if isinstance(dataloader, dict):
            self.dataloader = runner.build_dataloader(
                dataloader)
        else:
            self.dataloader = dataloader

    @property
    def runner(self):
        return self._runner

    @abstractmethod
    def run(self) -> Any:
        """Execute loop."""