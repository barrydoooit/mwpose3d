import threading
from typing import List, TYPE_CHECKING



if TYPE_CHECKING:
    from apps.pc_collection.runner import PcdCollectVisRunner
    from apps.pc_collection.gui.main_window import DataCollectorMainWindow
    from apps.pc_collection.onlineCollectLoop import OnlineDataCollectionLoop
    from apps.pc_collection.pc_buffer import PointCloudBuffer


class BaseCliCommand:
    def execute(self, runner: 'PcdCollectVisRunner', args: List[str]):
        raise NotImplementedError
    
class BufferSizeCommand(BaseCliCommand):
    def execute(self, runner: 'PcdCollectVisRunner', args: List[str]):
        if len(args) != 1:
            return "Usage: buf <buffer_size>"
        try:
            size = int(args[0])
            runner.buffer.max_buffer_size = size
            return f"Buffer size set to {size}"
        except ValueError:
            return "Invalid buffer size"

class OutputDirCommand(BaseCliCommand):
    def execute(self, runner: 'PcdCollectVisRunner', args: List[str]):
        if len(args) != 1:
            return "Usage: out <output_dir>"
        runner.buffer.output_dir = args[0]
        return f"Output directory set to {args[0]}"

class MidBreakLenComand(BaseCliCommand):
    def execute(self, runner: 'PcdCollectVisRunner', args: List[str]):
        if len(args) != 1:
            return "Usage: mid <mid_break_len>"
        try:
            mid_break_len = int(args[0])
            runner.buffer.mid_break_sec = mid_break_len
            return f"Mid break length set to {mid_break_len}"
        except ValueError:
            return "Invalid mid break length"

class ModeCommand(BaseCliCommand):
    def execute(self, runner: 'PcdCollectVisRunner', args: List[str]):
        if runner.loop.running:
            return "Loop is already running. <Stop> first."
        if len(args) != 1:
            return f"Usage: mod <{'|'.join([m.value for m in runner.buffer.Mode])}>"
        mode_str = args[0]
        return runner.switch_mode(mode_str)

class StartCommand(BaseCliCommand):
    def execute(self, runner: 'PcdCollectVisRunner', args: List[str]):
        if runner.loop.running:
            return "Loop is already running. <Stop> first."
        runner.gui.reset_visuals()
        runner.loop.start()
        return f"Data collection loop started with interval {runner.loop.interval} sec. Current mode: {runner._mode.value.upper()}"

class StopCommand(BaseCliCommand):
    def execute(self, runner: 'PcdCollectVisRunner', args: List[str]):
        if not runner.loop.running:
            return "Loop is not running. <Start> first."
        threading.Thread(target=runner.loop.stop, daemon=True).start()
        threading.Thread(target=runner.buffer.clear, daemon=True).start()
        return "Data collection loop stopped."
    
COMMANDS = dict(
    buf=BufferSizeCommand(),
    out=OutputDirCommand(),
    mid=MidBreakLenComand(),
    mod=ModeCommand(),
    start=StartCommand(),
    stop=StopCommand()
)

class CommandProcessor:
    def __init__(self, 
                 runner: 'PcdCollectVisRunner',):
        self.runner = runner
    
    def process(self, command_line: str):
        parts = command_line.strip().split()
        if not parts:
            return ""
        cmd = parts[0]
        args = parts[1:]
        if cmd in COMMANDS:
            return COMMANDS[cmd].execute(self.runner, args)
        return f"Unknown command: {cmd}"