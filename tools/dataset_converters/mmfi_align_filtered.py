
from pathlib import Path
import shutil



def align_filtered_mmwave(dataset_root: Path, filtered_root: Path):
    dataset_root = Path(dataset_root)
    filtered_root = Path(filtered_root)

    for bin_path in filtered_root.rglob('*.bin'):
        rel_path = bin_path.relative_to(filtered_root)
        seq_dir = rel_path.parent # Exx/Sxx/Axx

        target_dir = dataset_root / seq_dir / 'mmwave_filtered'
        target_dir.mkdir(parents=True, exist_ok=True)

        shutil.move(bin_path, target_dir / bin_path.name)
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Align filtered mmwave .bin files into mmwave_filtered folders'
    )
    parser.add_argument(
        '-d', '--dataset_root', required=True,
        help='Path to the MMFi dataset root (E01/S01/A01 structure)'
    )
    parser.add_argument(
        '-f', '--filtered_root', required=True,
        help='Path to the filtered_mmwave folder (parallel E##/S##/A## structure)'
    )
    args = parser.parse_args()

    align_filtered_mmwave(Path(args.dataset_root), Path(args.filtered_root))