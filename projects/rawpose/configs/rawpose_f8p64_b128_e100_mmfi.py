_base_ = [
    '../../../configs/__base__/default_runtime.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'projects.mmmesh'], allow_failed_imports=False)

data_prefix = dict(
    pcd='mmwave_filtered',
    skel='skeleton'
)
data_root = './data/mmfi'
train_info = 'info_subj_train.pkl'
val_info = 'info_subj_val.pkl'
test_info = 'info_subj_val.pkl'

keypoints_involved=list(range(0, 17))

num_frames = 8
backup_frames = 5
total_frames = num_frames + backup_frames
point_cloud_size = 64
model = dict(
    type="MmMeshPredictor",
    point_cloud_size=point_cloud_size,
    frame_len=num_frames,
    criterion='MSELoss',
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
            fc_channels=[64, 16, 2],
            learnable_init_state=True,
        )
    ),
    anchor_module_cfg=dict(
        type="AnchorModule",
        anchor_cfg=dict(
            grouping_nsample=8,
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
            bidirectional=False,
            learnable_init_state=True,
        )
    ),
    fusion_module_cfg=dict(
        type="SimpleKpFusionHead",
        channels=[128, 128, len(keypoints_involved)*3],
    ),
    train_cfg=dict(
        warmup_frames=0,
    ),
    test_cfg=dict(
        serial=False,
    )
)

train_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=5,
        num_frames=num_frames,
        backup_frames=backup_frames,
        empty_frame_op='prev',
    ),
    dict(
        type='RandomFrameDrop',
        drop_prob=0.05,
        max_drop=5,
    ),
    dict(
        type='SequenceClip',
        mode='last',
        sequence_length=num_frames
    ),
    dict(
        type='SkeletonKeypointFilter',
        keypoints_involved=keypoints_involved,
    ),
    dict(
        type='SkeletonCoordinateTransform',
        tran_xyz=(0, -3.15, 0)
    ),
    dict(
        type='PointCloudCoordinateTransform',
        tran_xyz=(0, -3.15, 0),
    ),
    dict(
        type='RandomTransform',
        transform_prob=0.8,
        sigma_xyz=(0.02, 0.02, 0.02),
        max_d_xyz=(0.1, 0.1, 0.1)
    ),
    dict(
        type='PointDuplicator',
        target_num_points=point_cloud_size
    ),
    dict(
        type='PointSortAndClip',
        target_num_points=point_cloud_size,
        sort_dim=4,
        sort_order='desc'
    ),
    dict(
        type='NormalizePointAttr',
        attr_indices=(3, 4,),
        means=(-0.00047, 16.56169),
        stds=(0.79512, 3.88067)
    ),
    dict(
        type='AddRangeDimension',
        insert_idx=3,
    ),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=16,
    shuffle=True,
    drop_last=True,
    dataset=dict(
        type='MotionDataset',
        data_root=f"{data_root}",
        info_path=f"{data_root}/{train_info}",
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        sequence_length=total_frames,
        allow_pad_sequence=False,
        max_sequences=4
    )
)

optimizer_cfg = dict(
    type='AdamW',
    lr = 0.0005,
    weight_decay=0.01
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=25
)




val_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=5,
        num_frames=num_frames,
        backup_frames=backup_frames,
        empty_frame_op='prev',
    ),
    dict(
        type='SequenceClip',
        mode='last',
        sequence_length=num_frames
    ),
    dict(
        type='SkeletonKeypointFilter',
        keypoints_involved=keypoints_involved,
    ),
    dict(
        type='SkeletonCoordinateTransform',
        tran_xyz=(0, -3.15, 0)
    ),
    dict(
        type='PointCloudCoordinateTransform',
        tran_xyz=(0, -3.15, 0),
    ),
    dict(
        type='PointDuplicator',
        target_num_points=point_cloud_size
    ),
    dict(
        type='PointSortAndClip',
        target_num_points=point_cloud_size,
        sort_dim=4,
        sort_order='desc'
    ),
    dict(
        type='NormalizePointAttr',
        attr_indices=(3, 4,),
        means=(-0.00047, 16.56169),
        stds=(0.79512, 3.88067)
    ),
    dict(
        type='AddRangeDimension',
        insert_idx=3,
    ),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    shuffle=False,
    dataset=dict(
        type='MotionDataset',
        data_root=f"{data_root}",
        info_path=f"{data_root}/{val_info}",
        data_prefix=data_prefix,
        pipeline=val_pipeline,
        sequence_length=total_frames,
        allow_pad_sequence=False,
        max_sequences=1
    )
)
"""
MMFI_KEYPOINT_TYPE = {
        "SPINE_BASE": 0,
        "HIP_RIGHT": 1,
        "KNEE_RIGHT": 2,
        "ANKLE_RIGHT": 3,
        "HIP_LEFT": 4,
        "KNEE_LEFT": 5,
        "ANKLE_LEFT": 6,
        "SPINE_MID": 7,
        "SPINE_SHOULDER": 8,
        "NECK": 9,
        "HEAD": 10,
        "SHOULDER_LEFT": 11,
        "ELBOW_LEFT": 12,
        "WRIST_LEFT": 13,
        "SHOULDER_RIGHT": 14,
        "ELBOW_RIGHT": 15,
        "WRIST_RIGHT": 16,
    }
"""

# MMFi connectivity
MMFI_CONNECTIVITY = {
    0: [1, 4, 7],
    1: [2],
    2: [3],
    4: [5],
    5: [6],
    7: [8],
    8: [9, 11, 14],
    9: [10],
    11: [12],
    12: [13],
    14: [15],
    15: [16]
}

# ad vis
visualizer_cfg = {
    "keypoints_involved": keypoints_involved,
    "keypoint_for_stats": [14,15,16],   # right shoulder, elbow, wrist
    "error_type": "abs_error",
    "window_size": 100,
    "follow": True,
    "max_points_per_frame": point_cloud_size,
    "dataset_connectivity": MMFI_CONNECTIVITY,
}
# add vis

metric=dict(
    type='SimpleGTPredAnalyzer',
    keypoints_involved=keypoints_involved,
    visualizer_cfg=visualizer_cfg,
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
        data_root=f"{data_root}",
        info_path=f"{data_root}/{test_info}",
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        sequence_length=total_frames,
        allow_pad_sequence=False
    )
)

test_cfg = dict(
    type='TestLoop',
    metric_cfg=metric,
    checkpoints=list(range(10, 101, 10)),
)

