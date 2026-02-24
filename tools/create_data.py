"""
uv run .\tools\create_data.py custom --root-path "E:\projects\mwpose3d\apps\impl\dataset_collection\traces\raw_new_pc" --out-dir "."
"""

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

def milipoint_data_prep(root_path: str,
                        out_dir: str,
                        partitions: dict[str, float] = {'train': 0.8, 'val': 0.1, 'test': 0.1},
                        seed: int = 20,
                        num_keypoints: int = 18):
    from tools.dataset_converters.milipoint_converter import MilipointDatasetConverter
    converter = MilipointDatasetConverter(
        input_root=Path(root_path),
        output_root=Path(out_dir),
        partitions=partitions,
        seed=seed,
        num_keypoints=num_keypoints)
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
    elif args.dataset == 'milipoint':
        partitions = input("Enter the split ratios as 'train:val:test' (empty for default: 0.8:0.1:0.1): ")
        if not partitions:
            partitions = {'train': 0.8, 'val': 0.1, 'test': 0.1}
        else:
            train, val, test = map(float, partitions.split(':'))
            partitions = {'train': train, 'val': val, 'test': test}
        seed = int(input("Enter the random seed (default: 20): ") or 20)
        num_keypoints = int(input("Enter the number of keypoints (9 or 18, default: 18): ") or 18)
        milipoint_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            partitions=partitions,
            seed=seed,
            num_keypoints=num_keypoints
        )
    elif args.dataset == 'paw':
        from tools.dataset_converters.paw_converter import AsteriosPawDatasetConverter
        converter = AsteriosPawDatasetConverter(
            input_root=args.root_path,
            output_root=args.out_dir,
            seed=42)
        converter.process_all()
    else:
        print(f"Processing Custom dataset: {args.dataset}")
        custom_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            use_gui=True)
        
if __name__ == '__main__':
    main()