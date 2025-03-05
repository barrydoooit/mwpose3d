custom_imports = dict(
    imports=['dl_engine'], allow_failed_imports=False)

data_root = './data/neat/penguin'
train_info = 'info_train.pkl'
val_info = 'info_test.pkl'
test_info = 'info_test.pkl'

keypoint_involved=[0,1,2,3,4,5,6,8,9,10]
# keypoint_involved=[x for x in range(20) if x not in [7, 11, 15, 19]]
point_cloud_size = 32
model = dict(
    type="MmMeshPredictor",
    point_cloud_size=point_cloud_size,
    keypoints_involved=keypoint_involved,
    intensity_norm=(19.7624, 5.7237),
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
            xyz_range=[-0.9, -0.4, -0.9, 0.9, 0.2, 1.5],
            xyz_interval=[0.3, 0.3, 0.3]
        ),
        anchor_pointnet_cfg=dict(
            channels=[24+4+3, 32, 48, 64],
            kernel_size=1
        ),
        anchor_voxelnet_cfg=dict(
            channels=[64, 96, 128, 64],
            kernel_size=((3,3,3), (5,1,3),(3,1,3),),
        ),
        anchor_rnn_cfg=dict(
            input_size=64,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
    ),
    fusion_module_cfg=dict(
        type="SimpleKpFusionHead",
        channels=[128, 128, len(keypoint_involved)*3],
    )
)

num_frames = 20
train_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=5,
        num_frames=num_frames,
        load_all_skeletons=True,
    ),
    dict(
        type='CoordinateTransform',
        radar_tilt=5,
        kinect_tilt=5,
        pcd_tran=(0, -2, 0),
        skel_tran=(-0.37, 0, -2)
    ),
    dict(
        type='PointCloudRangeFilter',
        point_cloud_range=[-1.0, -1.0, -1.5, 1, 1.0, 2.0],
        load_pcd_dim=5
    ),
    dict(
        type='PointDuplicator',
        target_num_points=point_cloud_size
    ),
    dict(
        type='AddRangeDimension',
        insert_idx=3,
    ),
    dict(
        type='SkeletonKeypointFilter',
        keypoint_involved=keypoint_involved,
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
        info_path=f"{data_root}/{train_info}",
        pipeline=train_pipeline,
        sequence_length=num_frames,
        allow_pad_sequence=False
    )
)

optimizer_cfg = dict(
    type='AdamW',
    lr = 0.0005,
    weight_decay=0.01
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=300,
    val_interval=100
)

val_pipeline = train_pipeline
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    shuffle=False,
    dataset=dict(
        type='MotionDataset',
        data_root=f"{data_root}/h5",
        info_path=f"{data_root}/{val_info}",
        pipeline=val_pipeline,
        sequence_length=num_frames,
        allow_pad_sequence=False
    )
)
val_cfg = dict(
    type='ValLoop'
)
test_pipeline = train_pipeline
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    shuffle=False,
    dataset=dict(
        type='MotionDataset',
        data_root=f"{data_root}/h5",
        info_path=f"{data_root}/{test_info}",
        pipeline=test_pipeline,
        sequence_length=num_frames,
        allow_pad_sequence=False
    )
)
test_cfg = dict(
    type='TestLoop'
)