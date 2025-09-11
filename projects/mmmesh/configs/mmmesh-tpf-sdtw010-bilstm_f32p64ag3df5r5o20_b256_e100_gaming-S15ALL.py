_base_ = [
    '../../../configs/__base__/default_runtime.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'projects.mmmesh'], allow_failed_imports=False)

data_prefix = dict(
    pcd='mmwave',
    skel='skeleton'
)
data_root = './data/gaming/S15OTBPX05'
train_info = f'info_all.pkl'
val_info = f'info_armrisely.pkl'
test_info = f'info_s15otbp6vly05_test.pkl'

keypoints_involved=[0, 1, 2, 4, 5, 6, 8, 9, 10]

num_frames = 32
backup_frames = 16
stack_frames = 3
total_frames = num_frames + backup_frames
point_cloud_size = 64
y_offset = 0 # -2.0
z_offset = 1
x_tilt = -15 # 15 degrees downward
skel_x_tilt = -10
skel_x_offset = 0.2
model = dict(
    type="MmMeshPredictor",
    point_cloud_size=point_cloud_size,
    frame_len=num_frames,
    criterion=[
        dict(type="MSELoss", weight=1.0), 
        dict(type="SoftDTW", weight=0.1)
    ],
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
            bidirectional=True
        )
    ),
    anchor_module_cfg=dict(
        type="AnchorModule",
        anchor_cfg=dict(
            grouping_nsample=4,
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
            bidirectional=True,
            learnable_init_state=True,
        )
    ),
    fusion_module_cfg=dict(
        type="SimpleKpFusionHead",
        channels=[256, 128, len(keypoints_involved)*3],
    ),
)

tracking_discretize_resolution=(0.05, 0.05, 1)
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
        insert_idx=5,
    ),
    dict(
        type='StackPointCloudFrames',
        stack_size=stack_frames,
        inject_index=False,
        keep_structure=False,
    ),
    dict(
        type='DensityFilter',
        mode='threshold',
        min_points=5,
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
        type='SkeletonCoordinateTransform',
        rotate_xyz=(skel_x_tilt, 0, 0),
        tran_xyz=(skel_x_offset, y_offset, 0)
    ),
    dict(
        type='PointCloudCoordinateTransform',
        rotate_xyz=(x_tilt, 0, 0),
        tran_xyz=(0, y_offset, 0)
    ),
    dict(
        type='LoadTrackingRecords',
        tracker_name='RKFTracker',
        anchor_frame='nearestperframe',
        centroid_queue_len=num_frames,
        ignore_axis=[2],
        tracker_cfg='configs/apps/trackers/rkf.py',
    ),
    dict(
        type='RelativeCoordtoTrackingCentroid',
        discretize_resolution=tracking_discretize_resolution
    ),
    dict(
        type='RandomTransform',
        transform_prob=0.5,
        sigma_xyz=(0.02, 0.02, 0.1),
        max_d_xyz=(0.1, 0.1, 0.4)
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
        means=(0.0430, 9.4746),
        stds=(1.3270, 9.2075)
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
    max_epochs=100,
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
        insert_idx=5,
    ),
    dict(
        type='StackPointCloudFrames',
        stack_size=stack_frames,
        inject_index=False,
        keep_structure=False,
    ),
    dict(
        type='DensityFilter',
        mode='threshold',
        min_points=5,
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
        type='SkeletonCoordinateTransform',
        rotate_xyz=(skel_x_tilt, 0, 0),
        tran_xyz=(skel_x_offset, y_offset, 0)
    ),
    dict(
        type='PointCloudCoordinateTransform',
        rotate_xyz=(x_tilt, 0, 0),
        tran_xyz=(0, y_offset, 0)
    ),
    dict(
        type='LoadTrackingRecords',
        tracker_name='RKFTracker',
        anchor_frame='nearestperframe',
        centroid_queue_len=num_frames,
        ignore_axis=[2],
        tracker_cfg='configs/apps/trackers/rkf.py',
    ),
    dict(
        type='RelativeCoordtoTrackingCentroid',
        discretize_resolution=tracking_discretize_resolution
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
        means=(0.0430, 9.4746),
        stds=(1.3270, 9.2075)
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
        type='ExperimentalSmoother',
        smoother_type="gaussian_ema",
        keypoints_involved=keypoints_involved,
        idx_spine_shoulder_elbow_wrist_lr=[0, 1, 2, 4, 8, 5, 9, 6, 10],
        smoother_kwargs=dict(
            mu_pos       = 0.0,   # meters
            delta_pos    = 0.05,  # meters
            mu_ang       = 0.0,   # radians
            delta_ang    = 0.35,  # ~20 degrees
            lam_min      = 0.05,
            lam_max      = 0.95,
            combine_rule = 0,      # 0=min, 1=product, 2=weighted
            combine_alpha= 0.5     # only used for weighted
        )
    ),
    # dict(
    #     type='ExperimentalSmoother',
    #     keypoints_involved=keypoints_involved,
    #     idx_spine_shoulder_elbow_wrist_lr=[0, 1, 2, 4, 8, 5, 9, 6, 10],
    #     smoother_kwargs=dict(
    #         use_spine_cv= False,
    #         follow_raw_angle_when_unlocked= False,
    #         # EMA rates
    #         shoulder_offset_lambda = 0.2, 
    #         length_lambda = 0.05,
    #         spine_ema_lambda = 0.55,
    #         angle_ema_lambda = 0.85,
    #         # Spine lock via EMA radius
    #         spine_lock_radius_m = 0.1,
    #         spine_lock_in_frames = 20,
    #         spine_lock_out_frames = 2,
    #         # CV (spine) restart rules
    #         cv_restart_frames = 2,
    #         cv_pos_violate_tol_m = 0.05,
    #         # Angle lock band
    #         angle_tol_deg = 22.0,
    #         angle_unlock_out_frames = 2,
    #         angle_lock_in_frames = 10
    #     )
    # ),
    # dict(
    #     type='SavGolayFilter',
    #     window_length=7,
    #     polyorder=2,
    #     deriv=0,
    #     delta=0.055,      # 1 / 20 Hz
    #     mode='reflect',
    #     time_axis=0
    # )
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
            "max_points_per_frame": 128,        # max points to visualize per frame
        }
    ),
    postprocess=postprocess,
)