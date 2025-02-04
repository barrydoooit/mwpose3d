import argparse
import sys
import os



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
import debugpy
from apps.lateral_tracking.breakoutRunner import BreakoutRunner
# from apps.skeleton_estim.skeletonRunner import SktEstimRunner
import apps.lateral_tracking.constants as lateral_const
# import apps.skeleton_estim.constants as skeleton_const

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mmwave breakout application")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--type", type=str, default="breakout", help="Type of the application")
    args = parser.parse_args()
    if args.debug:
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")

    if args.type == "breakout":
        game_runner = BreakoutRunner(
            config_file_path=lateral_const.P_CONFIG_PATH,
            cli_port=lateral_const.P_CLI_PORT,
            data_port=lateral_const.P_DATA_PORT
        )
    # elif args.type == "skeleton":
    #     game_runner = SktEstimRunner(
    #             config_file_path=skeleton_const.P_CONFIG_PATH,
    #             cli_port=skeleton_const.P_CLI_PORT,
    #             data_port=skeleton_const.P_DATA_PORT
    #         )
    else:
        raise ValueError("Unknown type of the application")
    game_runner.start()