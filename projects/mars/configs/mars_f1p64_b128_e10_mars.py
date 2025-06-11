_base_ = [
    '../../../configs/__base__/default_runtime.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'projects.mars'], allow_failed_imports=False)

data_prefix = dict(
    pcd='mmwave',
    skel='skeleton'
)
data_root = './data/mars/woutlier'
train_info = 'info_train.pkl'
val_info = 'info_val.pkl'
test_info = 'info_test.pkl'

keypoints_involved=[i for i in range(0, 21) if i not in [7, 11]]

num_frames =1
backup_frames = 0
total_frames = num_frames + backup_frames
point_cloud_size=64
pcd_dim = 5
model = dict(
    type='MarsPredictor',
    point_cloud_size=point_cloud_size,
    in_channels=pcd_dim,
    input_size=(8, 8),
    keypoints_involved=keypoints_involved,
)

train_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=pcd_dim,
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
        tran_xyz=(0, -1.92, 0)
    ),
    dict(
        type='PointCloudCoordinateTransform',
        tran_xyz=(0, -1.92, 0),
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
        sort_dim=(0,1,2,),
        sort_order='asc',
        sort_effective=True
    ),
    dict(
        type='NormalizePointAttr',
        attr_indices=(3, 4,),
        means=(-0.00096, 43.60179),
        stds=(0.49076, 63.31943)
    ),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=4,
    shuffle=True,
    dataset=dict(
        type='MotionDataset',
        data_root=f"{data_root}",
        info_path=f"{data_root}/{train_info}",
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        sequence_length=total_frames,
        allow_pad_sequence=False
    ),
    pin_memory=True,
    prefetch_factor=10
)

optimizer_cfg = dict(
    type='Adam',
    lr = 0.001,
    betas=(0.5, 0.999)
)
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=0.0005, weight_decay=0.01),
# )

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,
    val_interval=50
)

val_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=pcd_dim,
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
        tran_xyz=(0, -1.92, 0)
    ),
    dict(
        type='PointCloudCoordinateTransform',
        tran_xyz=(0, -1.92, 0),
    ),
    dict(
        type='PointDuplicator',
        target_num_points=point_cloud_size,
    ),
    dict(
        type='PointSortAndClip',
        target_num_points=point_cloud_size,
        sort_dim=(0,1,2,),
        sort_order='asc',
        sort_effective=True
    ),
    dict(
        type='NormalizePointAttr',
        attr_indices=(3, 4,),
        means=(-0.00096, 43.60179),
        stds=(0.49076, 63.31943)
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
    metric_cfg=metric
)

custom_hooks = [
    dict(
        type='LatencyProfilingHook',
        out_file="exp_data/latency/3090/mars_f1p64_b128_e300_mars.json",
        subject_modules=[],
        include_full_forward=True,
    )
]