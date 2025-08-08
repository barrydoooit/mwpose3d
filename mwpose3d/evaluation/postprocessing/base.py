from typing import Callable, List, Optional, Sequence, Union
from mwpose3d.datasets.skel_data_sample import SkeletonDataSample
from mwpose3d.datasets.transforms.base import BaseTransform
from mwpose3d.registry import TRANSFORMS as POSTPROCESSING
from mmengine.dataset import Compose


@POSTPROCESSING.register_module()
class BasePostProcessing(BaseTransform):
    def transform(self, datasample: SkeletonDataSample) -> SkeletonDataSample:
        raise NotImplementedError("Subclasses should implement this method.")

class ComposePostProcess(Compose):
    def __init__(self, postprocesses: Optional[Sequence[Union[dict, Callable]]]):
        self.postprocesses: List[Callable] = []

        if postprocesses is None:
            postprocesses = []

        for postprocess in postprocesses:
            # `Compose` can be built with config dict with type and
            # corresponding arguments.
            if isinstance(postprocess, dict):
                postprocess = POSTPROCESSING.build(postprocess)
                if not callable(postprocess):
                    raise TypeError(f'postprocess should be a callable object, '
                                    f'but got {type(postprocess)}')
                self.postprocesses.append(postprocess)
            elif callable(postprocess):
                self.postprocesses.append(postprocess)
            else:
                raise TypeError(
                    f'postprocess must be a callable object or dict, '
                    f'but got {type(postprocess)}')

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply postprocesses sequentially.

        Args:
            data (dict): A result dict contains the data to postprocess.

        Returns:
           dict: Transformed data.
        """
        for t in self.postprocesses:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.postprocesses`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + '('
        for t in self.postprocesses:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string