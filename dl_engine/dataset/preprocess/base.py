from abc import ABC, abstractmethod
from mmengine import Registry

TRANSFORM = Registry('mytransform')

@TRANSFORM.register_module()
class BaseTransform(ABC):
    @abstractmethod
    def transform(self, input):
        raise NotImplementedError
    
    def __call__(self, *args, **kwds):
        return self.transform(*args, **kwds)