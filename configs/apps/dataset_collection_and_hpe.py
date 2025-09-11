_base_ = [
    './readers/6843base.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'apps.impl.online_estim_and_collection'], allow_failed_imports=False)
type = 'HPESimulCollectionPredictionApp'
vis_cfg = dict(
    type='OnlineSkeletonVisualizer',
    joint_cnxn=((0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7),
                (2, 8), (8, 9), (9, 10), (10, 11), 
                (2, 12), (12, 13), (13, 14), (14, 15), 
                (2, 16), (16, 17), (17, 18), (18, 19)),
    joint_indices=list(range(20))
)

instructions = dict(
    on_init=[
        *[dict(content=f"Capture Starts in {X} seconds.", duration=1, repeats=1) for X in range(5, 0, -1)]
    ], 
    on_start=[
        # *[step for _ in range(2) for step in [dict(content="Anchor Motions: Rise your RIGHT arm UPWARDS", duration=2, repeats=1),
        # dict(content="Anchor Motions: Put down your RIGHT arm", duration=2, repeats=1),
        # dict(content="Anchor Motions: Rise your LEFT arm UPWARDS", duration=2, repeats=1),
        # dict(content="Anchor Motions: Put down your LEFT arm", duration=2, repeats=1)]],
        dict(content="Capture in progress...", duration=1, repeats=1)
    ],
    on_stop=[
        *[dict(content=f"Capture Stopped. Waiting for next capture to start ({X}s)", duration=1, repeats=1) for X in range(10, 5, -1)]
    ]
)

# data_root = 'apps/impl/dataset_collection/traces'
data_root = 'apps/impl/dataset_collection/traces/userstudy'
buffer_cfg = dict(
    dump_dir=f'{data_root}/pointcloud',
    buffer_size=800
)

kinect_cfg = dict(
    kinect_mgr_cfg=dict(
        exe_path='apps/impl/dataset_collection/DumpKinectSkeleton/bin/Release/DumpKinectSkeleton.exe',
        output_dir=f'{data_root}/kinect',
        mode=['capture', 'control'],
    )
)

key_event_collector_cfg = dict(
    storage_dir=f"{data_root}/keyevents",
    screen_index=0,
    gif_path="apps/impl/online_estim_and_collection/dissatisfied.gif",  # Replace with a valid GIF path
)

hpe_model_cfg = './projects/mmmesh/configs/mmmesh-tpf-sdtw010-bilstm_f32p64ag3df5r5o20_b256_e100_gaming-S15ALL.py'

# load_from = './checkpoints/ptransv1_f5p64_b16_e10_mmfi.pth'
# load_from = './checkpoints/ptransv1_f5p64_b16_e10_custom2m/phase_1-epoch_10.pth'
# load_from = './checkpoints/d20otbyk05.pth'
load_from = './checkpoints/s15otball-df5r5o20.pth'
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