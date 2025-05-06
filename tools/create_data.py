import argparse
from pathlib import Path
from typing import List, Literal
import sys
import os

import debugpy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import traceback

def custom_data_prep(root_path: str,
                     out_dir: str,
                     use_gui: bool = True,):
    from tools.rawproc.data_manager.core import DataProcessorDelegate, DataProcessorGUI
    root_path = Path(root_path)
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    if use_gui:
        print('Using GUI')
        try:
            delegate = DataProcessorDelegate(root_path, out_dir)
            gui = DataProcessorGUI(delegate)
            gui.mainloop()
        except KeyboardInterrupt:
            print('Interrupted')
        except Exception as e:
            traceback.print_exc()
        finally:
            return
    else:
        print("Data not processed. Using GUI to manually process files.")
        # example_episode_name = "20250227_122654-122851_647"
        # example_episode = Episode(episode_name=example_episode_name, episode_length=600)
        # example_episode.load_pcd(root_path / 'radar')
        # example_episode.load_pcd_meta(root_path / 'radar' / 'meta')
        # example_episode.load_skeleton(root_path / 'kinect')
        # calibrated_episode = example_episode.calibrate_time()
        # aligned_episode = calibrated_episode.align_traces(use_interp_skel=True)
        # ToHdf5(alligned_episode=aligned_episode, output_dir=out_dir).save()

def mars_data_prep(root_path: str,
                   out_dir: str,
                   outlier_option: str = 'wo',
                   clip_size: int = 512):
    from tools.dataset_converters.mars_converter import MarsDatasetConverter
    converters: List[MarsDatasetConverter] = []
    if outlier_option == 'w' or outlier_option == 'both':
        root_path = Path(root_path) / 'woutlier',
        out_dir = Path(out_dir) / 'woutlier',
        converters.append(MarsDatasetConverter(
            input_root=root_path,
            output_root=out_dir,
            clip_size=clip_size))
    
    if outlier_option == 'wo' or outlier_option == 'both':
        converters.append(MarsDatasetConverter(
            input_root=Path(root_path) / 'wooutlier',
            output_root=Path(out_dir) / 'wooutlier',
            clip_size=clip_size))
    
    for converter in converters:
        converter.process_all()

def mmfi_data_prep(root_path: str,
                   out_dir: str,
                   modality: List[Literal['mmwave', 'mmwave_filtered']],
                   unify_coordinate: bool = False,
                   ):
    from tools.dataset_converters.mmfi_converter import MMFiDatasetConverter
    converter = MMFiDatasetConverter(
        input_root=Path(root_path),
        output_root=Path(out_dir),
        unify_coordinate=unify_coordinate,
        modality=modality)
    converter.process_all()

def main():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument('dataset', help='name of the ataset')
    parser.add_argument(
        '--root-path',
        type=str,
        help='specify the root path of the dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='specify the output directory')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    args = parser.parse_args()
    if args.debug:
        debugpy.listen(5678)
        print('Waiting for debugger attach')
        debugpy.wait_for_client()
    if args.dataset == 'custom':
        custom_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            use_gui=True)
    elif args.dataset == 'mars':
        outlier_option = input("Create dataset with outliers (w), without outliers (wo), or both (both)? ").strip().lower()
        while outlier_option not in ['w', 'wo', 'both']:
            print("Invalid option. Please choose 'w', 'wo', or 'both'")
            outlier_option = input("Create dataset with outliers (w), without outliers (wo), or both (both)? ").strip().lower()
        mars_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            outlier_option=outlier_option)
    elif args.dataset == 'mmfi':
        modality = input("Create dataset with mmwave (m), mmwave_filtered (mf), or both (both)? ").strip().lower()
        while modality not in ['m', 'mf', 'both']:
            print("Invalid option. Please choose 'm', 'mf', or 'both'")
            modality = input("Create dataset with mmwave (m), mmwave_filtered (mf), or both (both)? ").strip().lower()
        modality = ['mmwave', 'mmwave_filtered'] if modality == 'both' else [{'m': 'mmwave', 'mf': 'mmwave_filtered'}[modality]]
        
        unify_coordinate = input("Unify coordinate (y/n)? ").strip().lower()
        while unify_coordinate not in ['y', 'n']:
            print("Invalid option. Please choose 'y' or 'n'")
            unify_coordinate = input("Unify coordinate (y/n)? ").strip().lower()
        unify_coordinate = unify_coordinate == 'y'
        mmfi_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            modality=modality,
            unify_coordinate=unify_coordinate)
if __name__ == '__main__':
    main()