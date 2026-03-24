import argparse
import sys
import os
import debugpy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



def main():
    # Stage 1: dataset name, debug flag, and which functionality to run.
    stage1 = argparse.ArgumentParser(
        description='Auxiliary data generation tool',
        add_help=False,
    )
    stage1.add_argument('dataset', help='name of the dataset')
    stage1.add_argument('--debug', action='store_true', help='enable debug mode')
    stage1.add_argument('--trec', action='store_true', help='generate tracking records')
    stage1.add_argument('--pointing', action='store_true',
                        help='generate pointing gesture annotations')
    stage1.add_argument('--pcd', action='store_true',
                        help='generate alternative point cloud variants from raw ADC data')
    stage1.add_argument('-h', '--help', action='store_true',
                        help='show this help message and exit')

    args1, remaining = stage1.parse_known_args()

    if args1.debug:
        debugpy.listen(5678)
        print('Waiting for debugger attach')
        debugpy.wait_for_client()

    if args1.help and not (args1.trec or args1.pointing or args1.pcd):
        stage1.print_help()
        print('\nPass a mode flag (--trec / --pointing / --pcd) together with -h '
              'for per-mode help.')
        sys.exit(0)

    # ------------------------------------------------------------------ trec
    if args1.trec:

        from tools.dataset_converters.generate_tracking_v2 import TrackingRecordGeneratorV2

        p = argparse.ArgumentParser(prog=f'{sys.argv[0]} {args1.dataset} --trec')
        p.add_argument('--hpe-cfg', type=str, help='hpe config file path')
        p.add_argument('--pcd-prefix', type=str, default='mmwave',
                       help='prefix for point cloud data folder')
        p.add_argument('--vis', action='store_true', default=False,
                       help='visualize the dataset')
        p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='dataset splits to process')
        trec_args = p.parse_args(remaining)

        trec_grt = TrackingRecordGeneratorV2(
            dataset=args1.dataset,
            hpe_cfg_f=trec_args.hpe_cfg,
            data_prefix=dict(pcd=trec_args.pcd_prefix),
            splits=trec_args.splits,
        )
        if trec_args.vis:
            trec_grt.visualize()
        else:
            trec_grt.generate()


    # ------------------------------------------------------------------- pcd
    if args1.pcd:
        from pathlib import Path
        from tools.dataset_converters.generate_pcd_variants import PcdVariantGenerator

        _DEFAULT_DSP_CFG_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs', 'dsp')
        
        p = argparse.ArgumentParser(prog=f'{sys.argv[0]} {args1.dataset} --pcd')
        p.add_argument('--data-dir', type=str, default=None,
                       help='formatted dataset directory (default: data/<dataset>)')
        p.add_argument('--raw-traces-dir', type=str, required=True,
                       help='raw traces directory containing raw/<episode>.bin files')
        p.add_argument('--dsp-cfg', nargs='+', type=str, default=None,
                       metavar='CFG_FILE',
                       help='headless mode: explicit list of DSP config .py files to apply')
        p.add_argument('--dsp-cfg-dir', type=str, default=None,
                       metavar='DIR',
                       help='GUI mode: open selector pre-populated from this directory '
                            f'(default: {_DEFAULT_DSP_CFG_DIR})')
        p.add_argument('--episodes', nargs='+', type=str, default=None,
                       metavar='EPISODE_ID',
                       help='restrict reprocessing to specific episode IDs')
        pcd_args = p.parse_args(remaining)

        data_dir = Path(pcd_args.data_dir or os.path.join('data', args1.dataset))

        gen = PcdVariantGenerator(
            dataset=args1.dataset,
            data_dir=data_dir,
            raw_traces_dir=Path(pcd_args.raw_traces_dir),
            dsp_cfg_paths=[Path(c) for c in pcd_args.dsp_cfg] if pcd_args.dsp_cfg else None,
            dsp_cfg_dir=Path(pcd_args.dsp_cfg_dir) if pcd_args.dsp_cfg_dir else None,
            episode_ids=pcd_args.episodes,
        )

        if pcd_args.dsp_cfg:
            gen.generate()
        else:
            gen.select_configs_and_generate()


if __name__ == '__main__':
    main()
