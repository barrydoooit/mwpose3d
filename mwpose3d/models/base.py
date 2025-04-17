from abc import abstractmethod
from typing import List, Optional, Tuple

from mwpose3d.datasets.skel_data_sample import SkeletonDataSample
from mwpose3d.registry import MODELS

import torch

@MODELS.register_module()
class BaseSkeletonEstimModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, 
                inputs: torch.Tensor,
                data_samples: Optional[List[SkeletonDataSample]] = None,
                mode: str = 'tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            return self._forward(inputs, data_samples)
        
    @abstractmethod
    def loss(self, batch_inputs: torch.Tensor, 
             data_samples: List[SkeletonDataSample]) -> torch.Tensor:
        pass
    
    @abstractmethod
    def predict(self, batch_inputs: torch.Tensor,
                data_samples: List[SkeletonDataSample]) -> torch.Tensor:
        pass
    
    @abstractmethod
    def _forward(self, batch_inputs: torch.Tensor,
                 data_samples: List[SkeletonDataSample]) -> torch.Tensor:
        pass

    @abstractmethod
    def pack_input(self, data_batch_dict: dict) -> Tuple[torch.Tensor, List[SkeletonDataSample]]:
        pass