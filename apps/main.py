import argparse
import sys
import os
import debugpy
from mmengine.config import Config, DictAction


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
from apps.lateral_tracking.breakoutRunner import BreakoutRunner
# from apps.skeleton_estim.skeletonRunner import SktEstimRunner
from apps.pc_collection.runner import PcdCollectVisRunner
from apps.simpler_reader.runner import SimpleReaderRunner
from apps.online_skeleton_estim.runner import OnlineSkeletonEstimationRunner
# import apps.skeleton_estim.constants as skeleton_const

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mmwave breakout application")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    if args.debug:
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    app_type = cfg.pop("type")
    if app_type is None:
        raise ValueError("Application type is not specified in the configuration file")
    print(cfg)
    if app_type == "breakout":
        runner = BreakoutRunner.from_cfg(cfg)
    # elif args.type == "skeleton":
    #     game_runner = SktEstimRunner.from_cfg(cfg)
    elif app_type == "cvis":
        runner = PcdCollectVisRunner.from_cfg(cfg)
    elif app_type == "simple":
        runner = SimpleReaderRunner.from_cfg(cfg)
    elif app_type == "infengine":
        runner = OnlineSkeletonEstimationRunner.from_cfg(cfg)
    else:
        raise ValueError("Unknown type of the application")
    
    runner.start()