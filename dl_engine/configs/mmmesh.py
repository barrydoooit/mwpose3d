custom_imports = dict(
    imports=['dl_engine'], allow_failed_imports=False)

data_root = './data/neat'


model = dict(
    type="MmMeshPredictor",
    base_pointnet_cfg=dict(
        type="BasePointNet",
        channels=[6, 8, 16, 24],
        kernel_size=1
    ),
    global_module_cfg=dict(
        type="GlobalModule",
        global_pointnet_cfg=dict(
            channels=[24+4, 32, 48, 64],
            kernel_size=1
        ),
        global_rnn_cfg=dict(
            in_channel=64,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
            dropout=0.1,
            fc_channels=[64, 16, 2]
        )
    ),
    anchor_module_cfg=dict(
        type="AnchorModule",
        anchor_cfg=dict(
            grouping_nsample=8,
            xyz_range=[-0.3, -0.3, -0.3, 0.3, 0.3, 2.1],
            xyz_interval=[0.3, 0.3, 0.3]
        ),
        anchor_pointnet_cfg=dict(
            channels=[24+4+3, 32, 48, 64],
            kernel_size=1
        ),
        anchor_voxelnet_cfg=dict(
            channels=[64, 128, 256, 512],
            kernel_size=((3,3,3), (5,1,1),(3,1,1),),
        ),
        anchor_rnn_cfg=dict(
            input_size=64,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
    )
)

num_frames = 10
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
        pipeline=train_pipeline,
        sequence_length=num_frames,
        allow_pad_sequence=False
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