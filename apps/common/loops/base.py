import threading
import time


class BaseRadarProcessLoop:
    def __init__(self, interval: float):
        self._interval = interval
        self._running = False
        self._thread = None
        self._stop_event = threading.Event()
    
    def mainloop(self):
        while not self._stop_event.is_set():
            ts_start = time.perf_counter()
            data = self._generate_data()
            ret = self._process_data(data)
            ts_end = time.perf_counter()
            time.sleep(max(0, self._interval - (ts_end - ts_start)))
        self._running = False

    @property
    def running(self):
        return self._running
    
    @property
    def interval(self):
        return self._interval
    
    def _generate_data(self):
        raise NotImplementedError
    
    def _process_data(self):
        raise NotImplementedError
    
    def start(self):
        if not self._running:
            self._before_start_hook()
            self._stop_event.clear()
            self._thread = threading.Thread(target=self.mainloop, daemon=True)
            self._thread.start()
            self._running = True
            self._after_start_hook()
    
    def stop(self):
        if self._running:
            self._before_stop_hook()
            self._stop_event.set()
            if threading.current_thread() != self._thread:
                self._thread.join()
            self._running = False
            self._after_stop_hook()
    
    def _before_start_hook(self):
        pass
    
    def _after_start_hook(self):
        pass
    
    def _before_stop_hook(self):
        pass
    
    def _after_stop_hook(self):
        pass