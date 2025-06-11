import argparse
import os.path as osp
import subprocess
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..'))

import debugpy


def sync(source_dir: str = ".", destination_dir: str = "/tudelft.net/staff-umbrella/phdvault/mwpose3d", destination_host: str = "daic"):
    destination_path: str = destination_host + ":" + destination_dir if destination_host else destination_dir
    print("Syncing local files with DAIC...")

    print("Testing whether rsync exists")
    subprocess.run(["rsync", "--version"], check=True)

    print("Creating destination directory")
    if len(destination_path) > 0:
        subprocess.run(["ssh", destination_host, "mkdir", "-p", destination_dir], check=True)
    else:
        subprocess.run(["mkdir", "-p", destination_path], check=True)

    print("Syncing files to destination")
    # Add HostName and User to ~/.ssh/config to support `daic:` syntax. Otherwise, use netid@daic.hostname.nl
    subprocess.run(["rsync",
                    "-rtD",                     # Recursive, preserve times, preserve special and device files (Cannot use --archive: https://unix.stackexchange.com/questions/558235/rsync-operation-not-permitted)
                    "--verbose",                # Print some statistics at the end (number of bytes, speedup, etc.)
                    "--compress",               # Compress data where possible, increases CPU, decreases bandwidth
                    r"--exclude=.*/",           # Exclude folders starting with a dot (.venv, .git, etc.)
                    r"--exclude=__pycache__/"   # Exclude python cache files
                    "--progress",               # Print progress per file
                    source_dir,                 # The local directory to synchronize, should be the root of the mwpose3d folder.
                    destination_path],          # The remote destination folder
                   check=True)

    print("Finished sync.")

def run_job():
    print("Submitting job to DAIC...")
    subprocess.run(["ssh", "daic", "sbatch train_mars.sbatch"], check=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Interface with the DAIC')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--sync', action='store_true', help='Synchronize local files with DAIC')
    parser.add_argument("--run-job", action='store_true', help='Submit a slurm job')

    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    if args.debug:
        debugpy.listen(5678)
        print('Waiting for debugger attach')
        debugpy.wait_for_client()

    if args.sync:
        sync()
    if args.run_job:
        run_job()


if __name__ == '__main__':
    main()
