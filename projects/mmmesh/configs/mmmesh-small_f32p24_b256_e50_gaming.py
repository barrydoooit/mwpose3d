_base_ = [
    '../../../configs/__base__/default_runtime.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'projects.mmmesh'], allow_failed_imports=False)

data_prefix = dict(
    pcd='mmwave',
    skel='skeleton'
)
data_root = './data/gaming'
train_info = 'info_train.pkl'
val_info = 'info_test.pkl'
test_info = 'info_test.pkl'

keypoints_involved=[0, 1, 2, 4, 5, 6, 8, 9, 10]

num_frames = 32
backup_frames = 5
total_frames = num_frames + backup_frames
point_cloud_size = 24
y_offset = 0 # -2.0
model = dict(
    type="MmMeshPredictor",
    point_cloud_size=point_cloud_size,
    frame_len=num_frames,
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
            fc_channels=[64, 16, 2],
            learnable_init_state=True,
        )
    ),
    anchor_module_cfg=dict(
        type="AnchorModule",
        anchor_cfg=dict(
            grouping_nsample=8,
            xyz_range=[-0.6, -0.3, -0.9, 0.6, 0.3, 1.5],
            xyz_interval=[0.6, 0.3, 0.3]
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
        type='AddRangeDimension',
        insert_idx=3,
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
        tran_xyz=(0, y_offset, -0.072)
    ),
    dict(
        type='PointCloudCoordinateTransform',
        tran_xyz=(0, y_offset, 0)
    ),
    dict(
        type='LoadTrackingRecords',
        tracker_name='RKFTracker',
        anchor_frame='firstlastthenfirstnext',
        ignore_axis=[2],
        translate=(0, y_offset, 0),
    ),
    dict(
        type='RelativeCoordtoTrackingCentroid',
        discretize_resolution=(0.05, 0.05, 1)
    ),
    dict(
        type='RandomTransform',
        transform_prob=0.5,
        sigma_xyz=(0.02, 0.02, 0.05),
        max_d_xyz=(0.1, 0.1, 0.2)
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
        means=(1.1662, -0.2264),
        stds=(0.3698, 0.7217)
    ),
]

train_dataloader = dict(
    batch_size=256,
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
    lr = 0.00025,
    weight_decay=0.01
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=10
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
        type='AddRangeDimension',
        insert_idx=3,
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
        tran_xyz=(0, y_offset, -0.072)
    ),
    dict(
        type='PointCloudCoordinateTransform',
        tran_xyz=(0, y_offset, 0)
    ),
    dict(
        type='LoadTrackingRecords',
        tracker_name='RKFTracker',
        anchor_frame='firstlastthenfirstnext',
        ignore_axis=[2],
        translate=(0, y_offset, 0),
        tracker_cfg='./configs/apps/trackers/rkf.py'
    ),
    dict(
        type='RelativeCoordtoTrackingCentroid',
        discretize_resolution=(0.075, 0.075, 1)
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
        means=(1.1662, -0.2264),
        stds=(0.3698, 0.7217)
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

postprocess = [
    dict(
        type='SkeletonBackToOriginalCoord',
    ),
    dict(
        type='SavGolayFilter',
        window_length=7,
        polyorder=2,
        deriv=0,
        delta=0.05,      # 1 / 20 Hz
        mode='reflect',
        time_axis=0
    )
]

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
    metric_cfg=dict(
        metric,
        visualizer_cfg={
            "keypoints_involved": keypoints_involved,          # same list you pass to analyzer
            "keypoint_for_stats": [5, 6],     # e.g., every other joint to reduce clutter
            "error_type": "abs_error",                # or "square_error"
            "window_size": 100,                       # frames in sliding window
            "follow": True,                           # auto-follow latest frame
            "max_points_per_frame": 60000,        # max points to visualize per frame
        }
    ),
    postprocess=postprocess,
)