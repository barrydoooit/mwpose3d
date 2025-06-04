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
                   use_official_split: bool = False,
                   clip_size: int = 512):
    from tools.dataset_converters.mars_converter import MarsDatasetConverter
    converters: List[MarsDatasetConverter] = []
    if outlier_option == 'pr':
        converters.append(MarsDatasetConverter(
            input_root=Path(root_path) / 'feature',
            output_root=Path(out_dir) / 'feature',
            use_processed_features=True,
            use_official_split=use_official_split,
            clip_size=clip_size))
        
    if outlier_option == 'w' or outlier_option == 'both':
        converters.append(MarsDatasetConverter(
            input_root=Path(root_path) / 'woutlier',
            output_root=Path(out_dir) / 'woutlier',
            use_official_split=use_official_split,
            clip_size=clip_size))
    
    if outlier_option == 'wo' or outlier_option == 'both':
        converters.append(MarsDatasetConverter(
            input_root=Path(root_path) / 'wooutlier',
            output_root=Path(out_dir) / 'wooutlier',
            use_official_split=use_official_split,
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

def mri_data_prep(root_path: str,
                  out_dir: str,
                  split: Literal[1, 2] = 2,
                  protocol: Literal[1, 2] = 2,
                  ratio: float = 0.8,
                  seed: int = 42,
                  ):
    from tools.dataset_converters.mri_converter import MRIDatasetConverter
    converter = MRIDatasetConverter(
        input_root=Path(root_path) / 'dataset_release' / 'aligned_data',
        output_root=Path(out_dir),
        split=split,
        protocol=protocol,
        ratio=ratio,
        seed=seed)
    converter.process_all()

def ask_for_option(prompt: str, options: List[str]) -> str:
    option = input(prompt).strip().lower()
    while option not in options:
        print(f"Invalid option. Please choose from {options}")
        option = input(prompt).strip().lower()
    return option

def main():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument('dataset', help='name of the dataset')
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
        outlier_option = ask_for_option(
            prompt="Create dataset with outliers (w), without outliers (wo), both (both), or use preprocessed features (pr)? ",
            options=['w', 'wo', 'both', 'pr'])
        official_split_option = ask_for_option(
            prompt="Use official split (y/n)? ",
            options=['y', 'n']) == 'y' if outlier_option != 'pr' else True
        mars_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            outlier_option=outlier_option,
            use_official_split=official_split_option)
    elif args.dataset == 'mmfi':
        modality = ask_for_option(
            prompt="Create dataset with mmwave (m), mmwave_filtered (mf), or both (both)? ",
            options=['m', 'mf', 'both']
        )
        modality = ['mmwave', 'mmwave_filtered'] if modality == 'both' else [{'m': 'mmwave', 'mf': 'mmwave_filtered'}[modality]]
        unify_coordinate = ask_for_option(
            prompt="Unify coordinate (y/n)? ",
            options=['y', 'n']) == 'y'
        mmfi_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            modality=modality,
            unify_coordinate=unify_coordinate)
    elif args.dataset == 'mri':
        use_defaults = ask_for_option(
            prompt="Use default settings (split=S2, protocol=P2, ratio=0.8, seed=42) (y/n)? ",
            options=['y', 'n']
        )
        if use_defaults == 'y':
            split, protocol, ratio, seed = 2, 2, 0.8, 42
        else:
            split = int(ask_for_option(
                prompt=(
                    "Choose data-split setting:\n"
                    "  1 (S1 Random Split) - random 80%/20% split of all samples\n"
                    "  2 (S2 Split by Subjects) - train on 80% of subjects, test on the rest\n"
                    "Enter 1 or 2: "
                ),
                options=['1', '2']
            ))
            protocol = int(ask_for_option(
                prompt=(
                    "Choose evaluation protocol:\n"
                    "  1 (P1) - all 12 movements (stretching, relaxing in free form, and walking)\n"
                    "  2 (P2) - only the first 10 rehabilitation movements\n"
                    "Enter 1 or 2: "
                ),
                options=['1', '2']
            ))
            ratio = float(input("Enter the train/val ratio (default: 0.8): ") or 0.8
            )
            seed = int(input("Enter the random seed (default: 42): ") or 42)

        mri_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            split=split,
            protocol=protocol,
            ratio=ratio,
            seed=seed
        )

if __name__ == '__main__':
    main()