_base_ = [
    '../../../configs/__base__/default_runtime.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'projects.radhar'], allow_failed_imports=False)

data_prefix = dict(
    pcd='mmwave_filtered',
    skel='skeleton'
)
data_root = './data/mmfi'
train_info = 'info_subj_train.pkl'
val_info = 'info_subj_val.pkl'
test_info = 'info_subj_val.pkl'

keypoints_involved=list(range(0, 17))

num_frames = 32
backup_frames = 5
total_frames = num_frames + backup_frames
pcd_dim = 5
point_cloud_range = [-1.2, 1.0, -1.5, 1.2, 4.5, 3.0]
voxel_size = [0.05, 0.05, 0.05]
sparse_shape = [
                    round((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
                round((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
                round((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),]
model = dict(
    type='RadHARCNNBiLSTM',
    num_frames=num_frames,
    voxel_size = voxel_size,
    point_cloud_range = point_cloud_range,
    moddle_encoder=dict(
        type='SparseEncoder',
        in_channels=pcd_dim,
        sparse_shape=sparse_shape,
        output_channels=64,
        encoder_channels=((16,), (32, 32), (32, 64), (64, 64)),
        encoder_paddings=((1,), (1, 1), (1, 1), (1, 1)),
        output_kernel_size=(3, 3, 3),
        output_stride=(2, 2, 2),
        output_padding=(1, 1, 1),
        block_type='conv_module',
        rulebook_reuse=True
    ),
    backbone=dict(
        type='SECOND',
        in_channels=384,
        out_channels=[128, 128],
        layer_nums=[3, 3],
        layer_strides=[1, 2],
    ),
    lstm_cfg=dict(
        input_size=768,
        hidden_size=64,
        num_layers=3,
        batch_first=True,
        dropout=0.1,
        bidirectional=True,
        learnable_init_state=True,
    ),
    head_channels=(128, 256, 128),
    keypoints_involved=keypoints_involved,
    criterion='sdtw',
)

train_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=5,
        num_frames=num_frames,
        backup_frames=backup_frames,
        empty_frame_op='zero'
    ),
    dict(
        type='PointCloudRangeFilter',
        point_cloud_range=point_cloud_range,
        empty_frame_op='shift',
        backup_frames=backup_frames,
        min_num_frames=num_frames,
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
        type='RandomTransform',
        transform_prob=0.8,
        sigma_xyz=(0.02, 0.02, 0.02),
        max_d_xyz=(0.1, 0.1, 0.1)
    ),
    dict(
        type='NormalizePointAttr',
        attr_indices=(3, 4,),
        means=(-0.00047, 16.56169),
        stds=(0.79512, 3.88067)
    ),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=16,
    shuffle=True,
  #  drop_last=True,
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
    lr = 0.00025,
    weight_decay=0.01
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=10
)


val_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=5,
        num_frames=num_frames,
        backup_frames=backup_frames,
        empty_frame_op='zero'
    ),
    dict(
        type='PointCloudRangeFilter',
        point_cloud_range=point_cloud_range,
        empty_frame_op='shift',
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
        type='NormalizePointAttr',
        attr_indices=(3, 4,),
        means=(-0.00047, 16.56169),
        stds=(0.79512, 3.88067)
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
        subject_modules=[],
        include_full_forward=True,
    )
]