import signal
import sys
import time
from typing import List
import serial
from mmengine.config import Config
from apps.online_skeleton_estim.OnlineReadPredictLoop import OnlineReaderPredictLoop
from dl_engine.runner.inference_engine import InferenceEngine
from dl_engine.runner.runner import ConfigType
from radario.base import BaseBufferedReader, build_reader
from radario.chirpConfig.chirpConfigIWR6843 import ChirpConfigIWR6843


class OnlineSkeletonEstimationRunner:
    def __init__(self,
                 reader_cfg: dict,
                 loop_cfg: dict,
                 deep_model_cfg_path: str,
                 checkpoint_path: str,
                 visualize: bool = False):
        self.reader = self._make_reader(reader_cfg)
        self.inference_engine = self._make_inference_engine(deep_model_cfg_path, checkpoint_path)
        self.loop = self._make_loop(loop_cfg, visualize)
    
    def _make_inference_engine(self, cfg_path: str, checkpoint_path: str) -> InferenceEngine:
        cfg = Config.fromfile(cfg_path)
        cfg.load_from = checkpoint_path
        return InferenceEngine.from_cfg(cfg)
    
    def _make_reader(self, reader_cfg: dict):
        cli_port = reader_cfg.get("CLI_port")
        cli_port = serial.Serial(cli_port, BaseBufferedReader.CLI_BAUDRATE)
        assert reader_cfg.get("type") == "BufferedPcdReaderIWR6843"
        chirp_cfg = ChirpConfigIWR6843(reader_cfg.pop("config_file_path"), CLI_port=cli_port)
        chirp_cfg.send_config(False)
        reader_cfg.update(dict(
            CLI_port=cli_port,
            Data_port=serial.Serial(reader_cfg.get("Data_port"), BaseBufferedReader.DATA_BAUDRATE, timeout=1.0)
        ))
        reader = build_reader(**reader_cfg)
        reader.register_config(chirp_cfg)
        return reader
        
    def _make_loop(self, loop_cfg: dict, visualize: bool):
        loop_cfg.update(dict(
            reader=self.reader,
            inference_engine=self.inference_engine,
            visualize=visualize
        ))
        return OnlineReaderPredictLoop.from_dict(loop_cfg)

    @classmethod
    def from_cfg(cls, cfg: ConfigType):
        return cls(
            reader_cfg=cfg.get("reader_cfg"),
            loop_cfg=cfg.get("loop_cfg"),
            visualize=cfg.get("visualize", False),
            deep_model_cfg_path=cfg.get("deep_model_cfg_path"),
            checkpoint_path=cfg.get("checkpoint_path")
        )
    
    def start(self):
        def sigint_handler(signum, frame):
            print("SIGINT received. Stopping loop...")
            self.loop.stop()
            sys.exit(0)
        signal.signal(signal.SIGINT, sigint_handler)

        self.loop.start()
        if self.loop.visualize:
            self.loop.root.mainloop()
            self.loop.stop()
        else:
            try:
                while True:
                    if sys.platform == "linux" or sys.platform == "darwin":
                        signal.pause()
                    else:
                        time.sleep(1)
                    
            except KeyboardInterrupt:
                print("KeyboardInterrupt captured.")
                self.loop.root.focus()
                self.loop.stop()
                sys.exit(0)