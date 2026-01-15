_base_ = [
    './readers/6843base.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'apps.impl.online_skeleton_estim'], allow_failed_imports=False)
type = 'OnlineSkeletionEstimationApp'
# hpe_model_cfg = './projects/mmmesh/configs/mmmesh-tpf-sdtw010-bilstm_f32p64ag3df5r5o20_b256_e100_gaming-S15ALL.py'
hpe_model_cfg = './projects/mmmesh/configs/mmmesh_f32p64_b128_e100_mmfi.py'


# load_from = './checkpoints/ptransv1_f5p64_b16_e10_mmfi.pth'
# load_from = './checkpoints/ptransv1_f5p64_b16_e10_custom2m/phase_1-epoch_10.pth'
# load_from = './checkpoints/d20otbyk05.pth'
# load_from = './checkpoints/s15otball-df5r5o20.pth'
load_from = './checkpoints/mmmesh_f32p64_b128_e100_mmfi.pth'

vis_cfg = dict(
    type='OnlineSkeletonVisualizer',
    # joint_cnxn=[[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], 
    #             [0, 7], [7, 8], [8, 9], [9, 10],
    #             [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]], # mmfi
    joint_cnxn=[[0, 1], [1, 2], [2, 3],
                [2, 4], [4, 5], [5, 6], [6, 7],
                [2, 8], [8, 9], [9, 10], [10, 11],
                [0, 12], [12, 13], [13, 14], [14, 15],
                [0, 16], [16, 17], [17, 18], [18, 19]], # kinect v2
    # joint_indices=list(range(0, 17)), 
    joint_indices=[0, 1, 2, 4, 5, 6, 8, 9, 10]
)