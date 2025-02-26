import time
from typing import Dict, Union
import serial
import logging

from radario.chirpConfig.chirpConfigIWR6843 import ChirpConfigIWR6843
from radario.readDataIWR6843 import BufferedPcdReaderIWR6843

from .utils import normalize_data
from .tracking.Tracking import BatchedData
from .tracking.buffer import TrackBuffer
from radario.chirpConfig import ChirpConfigIWR1443
from radario.readDataIWR1443 import BufferedPcdReaderIWR1443
log = logging.getLogger(__name__)

from .backend.visualManager import VisualManager
import apps.lateral_tracking.constants as const

from mmengine.config import Config, ConfigDict
ConfigType = Union[Dict, Config, ConfigDict]



class BreakoutRunner:
    def __init__(self, config_file_path: str,
                 cli_port: str,
                 data_port: str):
        self._make_reader(config_file_path, cli_port, data_port)
        self.track_buffer = TrackBuffer()
        self.batch = BatchedData()
        self.visual_manager = VisualManager()
       
    
    def _make_reader(self, config_file_path: str, cli_port: str, data_port: str):
        cli_port = serial.Serial(cli_port, BufferedPcdReaderIWR1443.CLI_BAUDRATE)
        
        if const.CUR_DEVICE == "IWR1443":
            self.config = ChirpConfigIWR1443(config_file_path, cli_port)
            self.config.send_config(close_port=False)
            self.reader = BufferedPcdReaderIWR1443(cli_port, data_port)
        elif const.CUR_DEVICE == "IWR6843":
            self.config = ChirpConfigIWR6843(config_file_path, cli_port)
            self.config.send_config(close_port=False)
            self.reader = BufferedPcdReaderIWR6843(cli_port, serial.Serial(data_port, BufferedPcdReaderIWR1443.DATA_BAUDRATE, timeout=1.0))
        else:
            raise ValueError("Unknown device. Change in constants.py")
        self.reader.register_config(self.config)
        self.sleep_time = 0.001 * self.config.frame_periodicity
        
        
    def start(self):
        
            try:
                while True:
                    t0 = time.time()
                    
                    data_ok, _frame_number, det_obj = self.reader.read()
                    if data_ok:
                        now = time.time()
                        print(det_obj)
                        self.track_buffer.dt = now - self.track_buffer.t
                        self.track_buffer.t = now
                        effective_data = normalize_data(det_obj)
                        self.track_buffer.track(effective_data, self.batch)
                    
                    self.visual_manager.update(self.track_buffer)
                
                    t_code = time.time() - t0
                    t_sleep = max(0, self.sleep_time / 2 - t_code)
                    time.sleep(t_sleep)

            except KeyboardInterrupt:
                log.info("KeyboardInterrupt received. Stopping . . .")
            except Exception as e:
                log.error(f"Error: {e}")
                e.with_traceback()
            finally:
                if hasattr(self.reader, "close"):
                    self.reader.close()
                    log.info("Reader closed.")
                del self.reader    
    
    @classmethod
    def from_cfg(cls, cfg: ConfigType):
        return cls(
            config_file_path=cfg.get("config_file_path"),
            cli_port=cfg.get("CLI_port"),
            data_port=cfg.get("Data_port")
        )