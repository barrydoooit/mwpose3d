import queue
import threading
from collections import deque

class SafeDeque:
    def __init__(self, maxsize=0):
        """
        Initialize the OverwriteQueue.

        Args:
            maxsize (int): Maximum number of items allowed in the queue.
                           If maxsize <= 0, the queue size is infinite.
        """
        self.maxsize = maxsize
        self.queue = deque(maxlen=maxsize if maxsize > 0 else None)
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def put(self, item, block=True, timeout=None):
        """
        Put an item into the queue. If the queue is full, remove the oldest item.

        Args:
            item: The item to be added.
            block (bool): Whether to block if necessary. Ignored in this implementation since
                          we overwrite the oldest item when full.
            timeout (float): Maximum time to block. Ignored in this implementation.
        """
        with self.lock:
            if self.maxsize > 0 and len(self.queue) >= self.maxsize:
                # Overwrite the oldest item
                removed_item = self.queue.popleft()
                print(f"Queue full. Removed oldest item: {removed_item}")
            self.queue.append(item)
            self.not_empty.notify()

    def get(self, block=True, timeout=None):
        """
        Remove and return an item from the queue.

        Args:
            block (bool): Whether to block if necessary.
            timeout (float): Maximum time to block.

        Returns:
            The oldest item from the queue.

        Raises:
            queue.Empty: If the queue is empty and blocking is False or timeout occurs.
        """
        with self.not_empty:
            if not block and not self.queue:
                raise Exception("Queue is empty")
            start_time = None
            if timeout is not None:
                import time
                start_time = time.time()
            while not self.queue:
                if not block:
                    raise Exception("Queue is empty")
                if timeout is not None:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        raise queue.Empty()
                    self.not_empty.wait(remaining)
                else:
                    self.not_empty.wait()
            item = self.queue.popleft()
            self.not_full.notify()
            return item

    def qsize(self):
        """Return the size of the queue."""
        with self.lock:
            return len(self.queue)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        with self.lock:
            return len(self.queue) == 0

    def full(self):
        """Return True if the queue is full, False otherwise."""
        with self.lock:
            if self.maxsize <= 0:
                return False
            return len(self.queue) >= self.maxsize

    def clear(self):
        """Clear all items from the queue."""
        with self.lock:
            self.queue.clear()

    def __str__(self):
        """Return a string representation of the queue."""
        with self.lock:
            return f"OverwriteQueue({list(self.queue)})"