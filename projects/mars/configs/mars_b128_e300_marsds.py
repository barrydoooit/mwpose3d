_base_ = [
    '../../../configs/__base__/default_runtime.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'projects.mars'], allow_failed_imports=False)

data_prefix = dict(
    pcd='mmwave',
    skel='skeleton'
)
data_root = './data/mars/wooutlier'
train_info = 'info_train.pkl'
val_info = 'info_val.pkl'
test_info = 'info_test.pkl'

keypoint_involved=list(range(0, 18))#[0,1,4,5,6,8,9,10]

num_frames =3
backup_frames = 0
total_frames = num_frames + backup_frames
point_cloud_size=64
model = dict(
    type='MarsPredictor',
    point_cloud_size=point_cloud_size,
    keypoints_involved=keypoint_involved,
    frame_len=num_frames,
)

train_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=5,
        num_frames=num_frames,
        backup_frames=backup_frames,
    ),
    dict(
        type='RandomTransform',
        transform_prob=0.5,
        sigma_xyz=(0.15, 0.15, 0.05),
        max_d_xyz=(0.3, 0.3, 0.1)
    ),
    dict(
        type='PointDuplicator',
        target_num_points=point_cloud_size,
    ),
    dict(
        type='PointSortAndClip',
        target_num_points=point_cloud_size,
        sort_dim=1, # 1 for Distance
        sort_order='asc'
    ),
    dict(
        type='SkeletonKeypointFilter',
        keypoint_involved=keypoint_involved,
        with_pcd_ts=False
    ),
    dict(
        type='NormalizePointAttr',
        attr_indices=(3, 4,),
        means=(0.00067, 43.26571),
        stds=(0.49076, 62.30057)
    )
]

train_dataloader = dict(
    batch_size=128,
    num_workers=16,
    shuffle=True,
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
    max_epochs=300,
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
        type='PointDuplicator',
        target_num_points=point_cloud_size
    ),
    dict(
        type='SkeletonKeypointFilter',
        keypoint_involved=keypoint_involved,
        with_pcd_ts=False
    ),
    dict(
        type='NormalizePointAttr',
        attr_indices=(3, 4,),
        means=(0.00067, 43.26571),
        stds=(0.49076, 62.30057)
    )
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
        data_root=f"{data_root}",
        info_path=f"{data_root}/{test_info}",
        data_prefix=data_prefix,
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

test_cfg = dict(
    type='TestLoop',
    metric_cfg=vis_metric
)