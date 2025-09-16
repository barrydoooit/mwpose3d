_base_ = [
    '../../../configs/__base__/default_runtime.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'projects.pointTS'], allow_failed_imports=False)

data_prefix = dict(
    pcd='mmwave_filtered',
    skel='skeleton_filtered'
)
data_root = './data/mri'
train_info = 'info_train_filtered.pkl'
val_info = 'info_val_filtered.pkl'
test_info = 'info_test_filtered.pkl'

keypoints_involved=[i for i in range(0, 17) if i not in [1,2,3,4]]
pointcloud_range = [-10, -5, -2, 10, 5, 2]
W=3
K=3
num_frames = K + W + 1
backup_frames = 10
total_frames = num_frames + backup_frames
point_cloud_size = 64
input_channels = 3
model = dict(
    type='PointTSPredictor',
    # Backbone builds a PointNet to extract per-window spatial features
    backbone_cfg=dict(
        type='PointNetBackbone',
        input_channels=input_channels,
        conv_channels=(128, 256, 512, 1024),
        global_feat_dim=256
    ),
    global_feat_dim=256,
    # Transformer for temporal encoding
    transformer_cfg=dict(
        num_layers=4,
        nhead=4,
        dim_feedforward=512,
        dropout=0.1,
        activation='relu',
        agg='last',
    ),
    keypoints_involved=keypoints_involved,
    point_cloud_size_per_frame=point_cloud_size,
    stacked_frames=W+1,
    input_channels=input_channels,
    train_cfg=dict(),
    test_cfg=dict(serial_test=True)
)

train_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=input_channels,
        num_frames=num_frames,
        backup_frames=backup_frames,
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
        tran_xyz=(0, -2.38, 0)
    ),
    dict(
        type='PointCloudCoordinateTransform',
        tran_xyz=(0, -2.38, 0),
    ),
    dict(
        type='PointCloudRangeFilter',
        point_cloud_range=pointcloud_range,
        empty_frame_op='shift',
        backup_frames=backup_frames,
        min_num_frames=num_frames
    ),
    dict(
        type='RandomTransform',
        transform_prob=0.8,
        sigma_xyz=(0.02, 0.02, 0.02),
        max_d_xyz=(0.1, 0.1, 0.1)
    ),
    dict(
        type='PointDuplicator',
        target_num_points=point_cloud_size,
    ),
    dict(
        type='PointSortAndClip',
        target_num_points=point_cloud_size,
        sort_dim=2,
        sort_order='desc'
    ),

    # dict(
    #     type='NormalizePointAttr',
    #     attr_indices=(3, 4,),
    #     means=(0.0, 28.98583),
    #     stds=(0.45029, 35.79703)
    # ),
    dict(
        type='StackPointCloudFrames',
        stack_size=W+1,
    ),
    dict(
        type='SequenceClip',
        mode='last',
        sequence_length=K+1
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
    max_epochs=120,
    val_interval=10
)


val_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=input_channels,
        num_frames=num_frames,
        backup_frames=backup_frames,
    ),
    dict(
        type='SkeletonKeypointFilter',
        keypoints_involved=keypoints_involved,
    ),
    dict(
        type='SkeletonCoordinateTransform',
        tran_xyz=(0, -2.38, 0)
    ),
    dict(
        type='PointCloudCoordinateTransform',
        tran_xyz=(0, -2.38, 0),
    ),
    dict(
        type='PointCloudRangeFilter',
        point_cloud_range=pointcloud_range,
        empty_frame_op='shift',
        backup_frames=backup_frames,
        min_num_frames=num_frames
    ),
    dict(
        type='PointDuplicator',
        target_num_points=point_cloud_size,
    ),
    dict(
        type='PointSortAndClip',
        target_num_points=point_cloud_size,
        sort_dim=2,
        sort_order='desc'
    ),
    # dict(
    #     type='NormalizePointAttr',
    #     attr_indices=(3, 4,),
    #     means=(0.0, 28.98583),
    #     stds=(0.45029, 35.79703)
    # ),
    dict(
        type='StackPointCloudFrames',
        stack_size=W+1,
    ),
    dict(
        type='SequenceClip',
        mode='last',
        sequence_length=K+1
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
        allow_pad_sequence=False
    )
)
metric=dict(
    type='SimpleGTPredAnalyzer',
    keypoints_involved=keypoints_involved,
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
)