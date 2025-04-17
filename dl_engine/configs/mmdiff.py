custom_imports = dict(
    imports=['dl_engine'], allow_failed_imports=False)

data_root = './data/neat/vital'
train_info = 'info_train.pkl'
val_info = 'info_test.pkl'
test_info = 'info_train.pkl'

keypoints_involved=[0,1,4,5,6,8,9,10]
num_joints = len(keypoints_involved)
point_cloud_size = 96
past_frames = 6
seq_frames = 5
num_frames = past_frames + seq_frames
backup_frames = 10
total_frames = num_frames + backup_frames
radar_input_c = 5
seq_tag = True
if seq_tag and seq_frames > 1:
    radar_input_c_mid = radar_input_c + 1
else:
    radar_input_c_mid = radar_input_c
    
model = dict(
    type="mmDiffPredictor",
    point_cloud_size=point_cloud_size,
    keypoints_involved=keypoints_involved,
    feature_extractor=dict(
        type="PointTransformerRegFeatureExtractor",
        input_dim=radar_input_c_mid,
        seq_tag=seq_tag,
        nblocks=5,
        n_p=num_joints,
        dropout=0.1,
        drop_key=0.1,
        dim=512, # 32 * 2^(nblocks-1)
        depth=5,
        dim_head=128,
        mlp_dim=256,
    ),
    diff_config=dict(
        hid_dim=96,
        emd_dim=96,
        coords_dim=[5,5], # NOTE: Not used
        num_layers=5,
        n_head=4,
        dropout=0.25,
        n_joints=num_joints,
    ),
    beta_cfg=dict(# For diffusion
        beta_schedule='linear',
        beta_start=0.0001,
        beta_end=0.001,
        num_diffusion_timesteps=51
    ),
    test_cfg=dict(
        skip_type='uniform',
        eta=0.0,
        test_times=1,
        test_timesteps=2,
        test_num_diffusion_timesteps=25,
        seq=range(0, 25, 25 // 2),
    ),
    global_feat_size=32,
    past_frames=past_frames,
    seq_frames=seq_frames,
    radar_input_c=radar_input_c,
    radar_input_c_mid=radar_input_c_mid,
    input_shape=(1, seq_frames, point_cloud_size, radar_input_c),
    output_shape=((1, num_joints, 3), (1, num_joints, 32)), # for feature extractor
    global_flag=1,
    local_flag=2,
    temp_flag=1,
    limb_flag=2,
    gt_norm_joint=1
)

train_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=radar_input_c,
        num_frames=seq_frames, # NOTE: This is for phase 1. Phase 2 will use num_frames
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
        duplicate_prob=0.
    ),
    dict(
        type='PointCloudRangeFilter',
        point_cloud_range=[-1.0, -1.0, -1.5, 1, 1.0, 2.0],
        empty_frame_op='shift',
        backup_frames=backup_frames
    ),
    dict(
        type='RandomFrameDrop',
        drop_prob=0.1,
        max_drop=2,
    ),
    dict(
        type='RandomScale',
        scale_prob=0.3,
        scale_range_x=(0.9, 1.1),
        scale_range_y=(0.95, 1.05),
        scale_range_z=(0.8, 1.1)
    ),
    dict(
        type='RandomRot3D',
        azimuth_range=(-5, 5),
        elevation_range=(-5, 5),
        roll_range=(-3, 3) 
    ),
    dict(
        type='RandomTransform',
        transform_prob=0.5,
        sigma_xyz=(0.15, 0.15, 0.1),
        max_d_xyz=(0.5, 0.5, 0.5)
    ),
    dict(
        type='SequenceClip',
        mode='last',
        sequence_length=seq_frames # NOTE: This is for phase 1. Phase 2 will use num_frames
    ),
    dict(
        type='SequenceReverse',
        reverse_prob=0.5,
        velocity_idx=3
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
        keypoint_involved=keypoints_involved,
        with_pcd_ts=False
    ),
    dict(
        type='NormalizePointAttr',
        attr_indices=(4,),
        means=(47.75,),
        stds=(81.46,)
    )
]

train_dataloader = dict(
    batch_size=16,
    num_workers=16,
    shuffle=True,
    dataset=dict(
        type='MotionDataset',
        data_root=f"{data_root}/h5",
        info_path=f"{data_root}/{train_info}",
        pipeline=train_pipeline,
        sequence_length=seq_frames + backup_frames,
        allow_pad_sequence=False
    )
)

val_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=radar_input_c,
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
        target_num_points=point_cloud_size,
        noise_std=0.
    ),
    dict(
        type='PointSortAndClip',
        target_num_points=point_cloud_size,
        sort_dim=1, # 1 for Distance
        sort_order='asc'
    ),
    dict(
        type='SkeletonKeypointFilter',
        keypoint_involved=keypoints_involved,
        with_pcd_ts=False
    ),
    dict(
        type='NormalizePointAttr',
        attr_indices=(4,),
        means=(47.75,),
        stds=(81.46,)
    )
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

lr_phase1 = 0.0001
lr_phase2 = 0.00002
optimizer_cfg = dict(
    type='AdamW',
    lr=lr_phase1,
    weight_decay=0.001
    )


train_cfg = dict(
    type='MMDiffTwoStageEpochBasedTrainLoop',
    pretrain_max_epochs=400,
    train_max_epochs=0,
    val_interval=20,
    load_pretrain_from="work_dirs/mmdiff/phase_1-epoch_380.pth",
    phase_cfg = dict(
        phase1=dict(
            amp=False,
            grad_clip=1.0,
        ),
        phase2=dict(
            ema=True,
            ema_rate=0.999,
            grad_clip=1.0,
            lr_decay_cfg=dict(
                interval=60,
                lr=lr_phase2,
                gamma=0.9
            ),
            lr=lr_phase2,
            num_frames=num_frames,
            backup_frames=backup_frames,
            batch_size=128,
        )
    )
)

metric=dict(
    type='SimpleGTPredAnalyzer',
    keypoint_involved=keypoints_involved,
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

test_cfg = dict(
    type='TestLoop',
    metric_cfg= dict(metric, visualizer_cfg=dict(
        keypoint_involved=keypoints_involved,
        keypoint_for_stats=[0, 5, 9],
        error_type='abs_error'
    ))
)
test_mode = 'coarse'
custom_hooks = [
    dict(type='MMDiffPipelineHook'),
]
