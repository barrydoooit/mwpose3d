custom_imports = dict(
    imports=['dl_engine'], allow_failed_imports=False)

data_root = './data/neat'

num_frames =3
model = dict(
    type='MarsPredictor',
    frame_len=num_frames,
)

train_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=5,
        num_frames=num_frames
    ),
    dict(
        type='PointCloudRangeFilter',
        point_cloud_range=[-1.0, 1.2, -1.5, 1, 2.4, 1.5]
    ),
    dict(
        type='SkeletonKeypointFilter',
        keypoint_involved=[x for x in range(20) if x not in [7, 11, 15, 19]],
        with_pcd_ts=False
    )
]

train_dataloader = dict(
    batch_size=128,
    num_workers=4,
    shuffle=True,
    dataset=dict(
        type='MotionDataset',
        data_root=f"{data_root}/h5",
        info_path=f"{data_root}/info_all.pkl",
        pipeline=train_pipeline
    )
)

optimizer_cfg = dict(
    type='AdamW',
    lr = 0.0001,
    weight_decay=0.01
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=200,
    val_interval=5
)

test_pipeline = train_pipeline
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    shuffle=False,
    dataset=dict(
        type='MotionDataset',
        data_root=f"{data_root}/h5",
        info_path=f"{data_root}/info_all.pkl",
        pipeline=test_pipeline
    )
)
test_cfg = dict(
    type='TestLoop'
)