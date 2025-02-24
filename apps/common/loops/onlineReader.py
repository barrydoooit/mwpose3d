import numpy as np
from radario.base import BaseBufferedReader
from .base import BaseRadarProcessLoop


class OnlineReaderLoop(BaseRadarProcessLoop):
    def __init__(self,
                 reader: BaseBufferedReader,
                 interval: float):
        super().__init__(interval)
        assert isinstance(reader, BaseBufferedReader), "Currently not supporting build the reader in the loop class. Build it outside and pass it in."
        self._reader = reader
    
    def _generate_data(self):
        data_ok, frame_number, det_obj = self._reader.read()
        return data_ok, frame_number, det_obj
    
    def _after_stop_hook(self):
        super()._after_stop_hook()
        if self._reader.Data_port.is_open:
            self._reader.Data_port.close()
    
    def _before_start_hook(self):
        super()._before_start_hook()
        if not self._reader.Data_port.is_open:
            self._reader.Data_port.open()
    
    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(
            reader=cfg.get("reader"),
            interval=cfg.get("interval")
        )

class TestingLoop(BaseRadarProcessLoop):
    def __init__(self,
                 reader: None,
                 interval: float):
        super().__init__(interval)
        self._reader = reader
    
    def _generate_data(self):
        num_points = np.random.randint(10, 50)
        det_obj = {"x": np.random.uniform(-2, 2, num_points),
                     "y": np.random.uniform(-2, 2, num_points),
                     "z": np.random.uniform(-2, 2, num_points),
                     "vel": np.random.uniform(-2, 2, num_points),
                     "snr": np.random.uniform(-2, 2, num_points)}
        return  1, 0, det_obj
