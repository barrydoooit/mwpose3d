from abc import ABC, abstractmethod
from mwpose3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class BaseTransform:
    def __init__(self, online_mode=False):
        self.online_mode = online_mode
    
    def transform(self, input):
        raise NotImplementedError

    def transform_online(self, input):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        if self.online_mode:
            return self.transform_online(*args, **kwargs)
        return self.transform(*args, **kwargs)