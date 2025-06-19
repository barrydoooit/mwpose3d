_base_ = [
    './readers/6843base.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'apps.impl.online_skeleton_estim'], allow_failed_imports=False)
type = 'OnlineSkeletionEstimationApp'
hpe_model_cfg = './projects/mmdiff/configs/ptransv1_f5p64_b16_e10_mmfi.py'
load_from = './checkpoints/ptransv1_f5p64_b16_e10_mmfi.pth'
vis_cfg = dict(
    type='OnlineSkeletonVisualizer',
    joint_cnxn=[[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], 
                [0, 7], [7, 8], [8, 9], [9, 10],
                [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]],
    joint_indices=list(range(0, 17)), 
)