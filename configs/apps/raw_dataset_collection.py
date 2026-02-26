_base_ = []

custom_imports = dict(
    imports=['mwpose3d', 'apps.impl.dataset_collection.raw_dataset_collection'], allow_failed_imports=False)

type = 'RawDatasetCollectionApp'

vis_cfg = dict(
    type='OnlineSkeletonVisualizer',
    joint_cnxn=((0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7),
                (2, 8), (8, 9), (9, 10), (10, 11), 
                (2, 12), (12, 13), (13, 14), (14, 15), 
                (2, 16), (16, 17), (17, 18), (18, 19)),
    joint_indices=list(range(20)),
    highlight_pointing=True
)

instructions = dict(
    on_init=[
        *[dict(content=f"Capture Starts in {X} seconds.", duration=1, repeats=1) for X in range(5, 0, -1)]
    ], 
    on_start=[
        dict(content="Capture in progress...", duration=1, repeats=1)
    ],
    on_stop=[
        *[dict(content=f"Capture Stopped. Waiting for next capture to start ({X}s)", duration=1, repeats=1) for X in range(5, 0, -1)]
    ]
)

data_root = 'apps/impl/dataset_collection/traces/try_1_raw'

reader_cfg = dict(
    type='UdpRawDataReader',
    process_point_cloud=True,
    save_to_file=f'{data_root}/raw/timed_frames.bin'
)

buffer_cfg = dict(
    dump_dir=f'{data_root}/pointcloud',
    buffer_size=5000
)

kinect_cfg = dict(
    kinect_mgr_cfg=dict(
        exe_path='apps/impl/dataset_collection/DumpKinectSkeleton/bin/Release/DumpKinectSkeleton.exe',
        output_dir=f'{data_root}/kinect',
        mode=['capture', 'control'],
    )
)
