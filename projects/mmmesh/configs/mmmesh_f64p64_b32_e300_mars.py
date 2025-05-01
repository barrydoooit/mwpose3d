_base_ = [
    '../../../configs/__base__/default_runtime.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'projects.mmmesh'], allow_failed_imports=False)

data_prefix = dict(
    pcd='mmwave',
    skel='skeleton'
)
data_root = './data/mars/wooutlier'
train_info = 'info_train.pkl'
val_info = 'info_val.pkl'
test_info = 'info_test.pkl'

keypoint_involved=list(range(0, 20))

num_frames = 64
backup_frames = 0
total_frames = num_frames + backup_frames
point_cloud_size = 64
model = dict(
    type="MmMeshPredictor",
    point_cloud_size=point_cloud_size,
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
        target_num_points=point_cloud_size
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
        means=(-0.99117, 33.58460),
        stds=(2.68049, 7.88112)
    ),
    dict(
        type='AddRangeDimension',
        insert_idx=3,
    ),
]

train_dataloader = dict(
    batch_size=32,
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
        means=(-0.99117, 33.58460),
        stds=(2.68049, 7.88112)
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

test_cfg = dict(
    type='TestLoop',
    metric_cfg=metric
)