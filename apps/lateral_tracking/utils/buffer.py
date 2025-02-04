import numpy as np
from collections import deque


class RingBuffer:
    """
    Buffer with a fixed size that automatically discards the oldest elements when new elements are added.

    Attributes
    ----------
    size : int
        Maximum size of the buffer.

    buffer : collections.deque
        Deque representing the circular buffer.

    Methods
    -------
    append(item)
        Add a new element to the buffer. If the buffer is full, the oldest element is removed.

    get_max()
        Get the maximum value in the buffer.

    get_mean()
        Get the mean value of the elements in the buffer.
    """

    def __init__(self, size, init_val=None):
        self.size = size
        self.buffer = deque(maxlen=size)
        if init_val is None:
            self.buffer.append(0)
        else:
            self.buffer.append(init_val)

    def append(self, item):
        self.buffer.append(item)

    def get_max(self):
        return np.max(self.buffer)

    def get_mean(self):
        return np.mean(self.buffer)