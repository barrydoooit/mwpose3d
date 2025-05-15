_base_ = [
    '../../../configs/__base__/default_runtime.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'projects.mars'], allow_failed_imports=False)

data_prefix = dict(
    pcd='mmwave',
    skel='skeleton'
)
data_root = './data/mri'
train_info = 'info_train.pkl'
val_info = 'info_val.pkl'
test_info = 'info_test.pkl'

keypoints_involved=[i for i in range(0, 17) if i not in [1,2,3,4]]

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
        tran_xyz=(0, -2.38, 0)
    ),
    dict(
        type='PointCloudCoordinateTransform',
        tran_xyz=(0, -2.38, 0),
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
        means=(0.0, 28.98583),
        stds=(0.45029, 35.79703)
    ),
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
    max_epochs=150,
    val_interval=25
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
        tran_xyz=(0, -2.38, 0)
    ),
    dict(
        type='PointCloudCoordinateTransform',
        tran_xyz=(0, -2.38, 0),
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
        means=(0.0, 28.98583),
        stds=(0.45029, 35.79703)
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