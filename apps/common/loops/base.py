import threading
import time
import traceback


class BaseRadarProcessLoop:
    def __init__(self, interval: float):
        self._interval = interval
        self._running = False
        self._thread = None
        self._stop_event = threading.Event()
    
    def mainloop(self):
        try:
            while not self._stop_event.is_set():
                ts_start = time.perf_counter()
                data = self._generate_data()
                ret = self._process_data(data)
                ts_end = time.perf_counter()
                time.sleep(max(0, self._interval - (ts_end - ts_start)))
                # print(f"Time taken: {ts_end - ts_start}")
        except KeyboardInterrupt:
            print("Keyboard interrupt")
        except Exception as e:
            print("Exception in loop:", e)
            traceback.print_exc()
        finally:
            self._running = False

    @property
    def running(self):
        return self._running
    
    @property
    def interval(self):
        return self._interval
    
    def _generate_data(self):
        return None
    
    def _process_data(self, data):
        pass
    
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
                self._thread.join(timeout=2.0)
                if self._thread.is_alive():
                    print("Thread did not stop within timeout. Continuing shutdown.")
            self._running = False
            self._thread = None
            self._after_stop_hook()
    
    @property
    def thread(self):
        
        return self._thread
    
    def _before_start_hook(self):
        pass
    
    def _after_start_hook(self):
        pass
    
    def _before_stop_hook(self):
        pass
    
    def _after_stop_hook(self):
        pass