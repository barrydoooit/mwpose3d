from enum import Enum
import sys
import threading
import time
from typing import Literal, Union
import tkinter as tk

import numpy as np
import serial
from apps.common.pcd.pointCloud import SimplePoint5D, SimplePointCloud5D
from apps.pc_collection.gui.cli import CommandProcessor
from apps.pc_collection.gui.main_window import DataCollectorMainWindow
from apps.pc_collection.onlineCollectLoop import OnlineDataCollectionLoop
from apps.pc_collection.pc_buffer import PointCloudBuffer
from radario.base import BaseBufferedReader, build_reader
from radario.chirpConfig.chirpConfigIWR6843 import ChirpConfigIWR6843


class PcdCollectVisRunner:
    class Mode(Enum):
        COLLECT = "collect"
        VISUALIZE = "visualize"
        
    def __init__(self,
                 buffer_cfg: dict,
                 reader_cfg: dict,
                 gui_cfg: dict,
                 loop_cfg: dict,
                 mode: Mode = Mode.COLLECT):
        self.reader = self._make_reader(reader_cfg)
        self.buffer = self._make_buffer(buffer_cfg)
        self.gui = self._make_gui(gui_cfg)
        self.loop = self._make_loop(loop_cfg)
        self._mode = mode
        self.cli_processor = CommandProcessor(self)
        self.gui.bind_cli_processor(self.cli_processor)
    
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
        
    def _make_buffer(self, buffer_cfg: dict):
        return PointCloudBuffer.from_dict(buffer_cfg)

    def _make_gui(self, gui_cfg: dict):
        return DataCollectorMainWindow.from_dict(gui_cfg)
    
    def _make_loop(self, loop_cfg: dict):
        loop_cfg.update(dict(
            reader=self.reader,
            buffer=self.buffer,
            gui=self.gui
        ))
        return OnlineDataCollectionLoop.from_dict(loop_cfg)
    
    def start(self):
        try:
            self.loop.start()
            self.gui.root.mainloop()
        except KeyboardInterrupt:
            print("Keyboard interrupt")
        except Exception as e:
            print(e)
        finally:
            self.loop.stop()
            sys.exit(0)
    
    def switch_mode(self, mode: Union[str, Mode]):
        if isinstance(mode, str):
            mode = self.Mode(mode.lower())
        self._mode = mode
        if mode == self.Mode.VISUALIZE:
            self.buffer.change_max_buffer_size(1)
        else:
            if self.buffer.max_buffer_size == 1:
                self.buffer.change_max_buffer_size(100)
        return f"Mode switched to {mode.value.upper()}"
        
    @classmethod
    def from_cfg(cls, cfg: dict):
        return cls(
            buffer_cfg=cfg.get("buffer_cfg"),
            reader_cfg=cfg.get("reader_cfg"),
            gui_cfg=cfg.get("gui_cfg"),
            loop_cfg=cfg.get("loop_cfg"),
            mode=cfg.get("mode", cls.Mode.VISUALIZE)
        )