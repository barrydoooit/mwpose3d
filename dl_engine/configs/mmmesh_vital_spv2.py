custom_imports = dict(
    imports=['dl_engine'], allow_failed_imports=False)

data_root = './data/neat/vital'
train_info = 'info_train.pkl'
val_info = 'info_test.pkl'
test_info = 'info_test.pkl'

keypoint_involved=[0,1,4,5,6,8,9,10]
# keypoint_involved=[x for x in range(20) if x not in [7, 11, 15, 19]]
point_cloud_size = 96
model = dict(
    type="MmMeshPredictor",
    point_cloud_size=point_cloud_size,
    keypoints_involved=keypoint_involved,
    sort_dim=1,
    sort_order='asc',
    intensity_norm=(43.75604688636036, 5.0),
    criterion='sdtw',
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
            # xyz_range=[-0.9, -0.4, -0.9, 0.9, 0.2, 1.5],
            # xyz_interval=[0.3, 0.1, 0.3]
            xyz_range=[-0.3, -0.3, -0.9, 0.3, 0.3, 1.5],
            xyz_interval=[0.3, 0.3, 0.3]
        ),
        anchor_pointnet_cfg=dict(
            channels=[24+4+3, 32, 48, 64],
            kernel_size=1
        ),
        anchor_voxelnet_cfg=dict(
            channels=[64, 96, 128, 64],
            # kernel_size=((3,3,3), (5,3,3),(3,3,3),),
            kernel_size=((3,3,3), (5,1,1), (3,1,1)),
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

num_frames = 64
backup_frames = 10
total_frames = num_frames + backup_frames
train_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=5,
        num_frames=num_frames,
        backup_frames=backup_frames,
    ),
    dict(
        type='CoordinateTransform',
        radar_tilt=5,
        kinect_tilt=5,
        pcd_tran=(0, -2, 0),
        skel_tran=(-0.37, 0, -2)
    ),
    dict(
        type='RandomFlip',
        flip_prob=0.5,
        duplicate_prob=0.1
    ),
    dict(
        type='PointCloudRangeFilter',
        point_cloud_range=[-1.0, -1.0, -1.5, 1, 1.0, 2.0],
        empty_frame_op='shift',
        backup_frames=backup_frames
    ),
    dict(
        type='RandomScale',
        scale_prob=0.1,
        scale_range_x=(0.95, 1.05),
        scale_range_y=(0.95, 1.05),
        scale_range_z=(0.9, 1.1)
    ),
    dict(
        type='RandomTransform',
        transform_prob=0.5,
        sigma_xyz=(0.15, 0.15, 0.05),
        max_d_xyz=(0.3, 0.3, 0.1)
    ),
    dict(
        type='SequenceClip',
        mode='last',
        sequence_length=num_frames
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
    ),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=16,
    shuffle=True,
    drop_last=True,
    dataset=dict(
        type='MotionDataset',
        data_root=f"{data_root}/h5",
        info_path=f"{data_root}/{train_info}",
        pipeline=train_pipeline,
        sequence_length=total_frames,
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
    max_epochs=400,
    val_interval=50
)


val_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=5,
        num_frames=num_frames,
        backup_frames=backup_frames,
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
        empty_frame_op='shift',
        backup_frames=backup_frames
    ),
    dict(
        type='SequenceClip',
        mode='last',
        sequence_length=num_frames
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
    ),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    shuffle=False,
    dataset=dict(
        type='MotionDataset',
        data_root=f"{data_root}/h5",
        info_path=f"{data_root}/{val_info}",
        pipeline=val_pipeline,
        sequence_length=total_frames,
        allow_pad_sequence=False
    )
)
metric=dict(
    type='SimpleGTPredAnalyzer',
    keypoint_involved=keypoint_involved,
)
val_cfg = dict(
    type='ValLoop',
    metric_cfg=metric
)
test_pipeline = val_pipeline
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    shuffle=False,
    dataset=dict(
        type='MotionDataset',
        data_root=f"{data_root}/h5",
        info_path=f"{data_root}/{test_info}",
        pipeline=test_pipeline,
        sequence_length=total_frames,
        allow_pad_sequence=False
    )
)

vis_metric = dict(metric, visualizer_cfg=dict(
        keypoint_involved=keypoint_involved,
        keypoint_for_stats=[0, 5, 9],
        error_type='abs_error'
    )
)

ana_window_size = 1
ana_metric = dict(
    type='PivotRotationAnalyzer',
    bones=((5,6), (9,10)),
    window_size_frames=ana_window_size,
    keypoint_involved=keypoint_involved,
    pos_pivot='first',
    output_dir='exp_data/test_logs',
    log_name=f'mmmesh_vital_spv2_pos_angle_norm_w{ana_window_size}.json'
)

res_metric = dict(
    type='ControlResolutionAnalyzer',
    keypoint_involved=keypoint_involved,
    controlled_keypoints=[5, 6, 9, 10],
    seg_length_n=1,
    seg_correct_threshold=0.8,
    motion_range_clip_ratio_xyz=[1.0, 0.8, 0.8],
    acc_guarantee_k=[0.7, 0.8, 0.9]
)

vol_metric = dict(
    type='VelocityEntropyAnalyzer',
    keypoint_involved=keypoint_involved,
    controlled_keypoints=[5, 9]
)
test_cfg = dict(
    type='TestLoop',
    metric_cfg=vis_metric
)