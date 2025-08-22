_base_ = [
    '../../../configs/__base__/default_runtime.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'projects.mmdiff'], allow_failed_imports=False)

data_prefix = dict(
    pcd='mmwave_filtered',
    skel='skeleton'
)
data_root = './data/mmfi'
train_info = 'info_subj_train.pkl'
val_info = 'info_subj_val.pkl'
test_info = 'info_subj_val.pkl'

keypoints_involved=list(range(0, 17))
num_joints = len(keypoints_involved)

point_cloud_size = 64
past_frames = 6
seq_frames = 10
num_frames = past_frames + seq_frames
backup_frames = 1
total_frames = num_frames + backup_frames
radar_input_c = 5
seq_tag = False
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
)

train_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=5,
        num_frames=seq_frames,
        backup_frames=backup_frames,
        empty_frame_op='prev',
    ),
    dict(
        type='RandomFrameDrop',
        drop_prob=0.05,
        max_drop=1,
    ),
    dict(
        type='SequenceClip',
        mode='last',
        sequence_length=seq_frames
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
]

train_dataloader = dict(
    batch_size=16,
    num_workers=16,
    shuffle=True,
    dataset=dict(
        type='MotionDataset',
        data_root=f"{data_root}",
        info_path=f"{data_root}/{train_info}",
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        sequence_length=seq_frames + backup_frames,
        allow_pad_sequence=False
    )
)

val_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=5,
        num_frames=seq_frames,
        backup_frames=backup_frames,
        empty_frame_op='prev',
    ),
    dict(
        type='SequenceClip',
        mode='last',
        sequence_length=seq_frames
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
        sequence_length=seq_frames + backup_frames,
        allow_pad_sequence=False
    )
)

lr_phase1 = 0.0001
lr_phase2 = 0.00002
optimizer_cfg = dict(
    type='AdamW',
    lr=lr_phase1,
    weight_decay=1e-8
    )


train_cfg = dict(
    type='MMDiffTwoStageEpochBasedTrainLoop',
    pretrain_max_epochs=10,
    train_max_epochs=0,
    val_interval=1,
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
        sequence_length=seq_frames + backup_frames,
        allow_pad_sequence=False
    )
)

test_cfg = dict(
    type='TestLoop',
    metric_cfg=metric
)
test_mode = 'coarse'
custom_hooks = [
    dict(type='MMDiffPipelineHook'),
    # dict(
    #     type='LatencyProfilingHook',
    #     out_file="/home/orin/Projects/mwpose3d/exp_data/latency/orin15/ptransv1_f1p64_b16_e10_mmfi.json",
    #     subject_modules=[],
    #     include_full_forward=True,
    # )
]
