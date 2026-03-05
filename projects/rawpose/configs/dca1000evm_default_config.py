_base_ = [
    '../configs/__base__/default_runtime.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'projects.mmmesh'], allow_failed_imports=False)

# Dataset path and prefix
data_root = 'tmp/try_1'
train_info = 'info_all.pkl'

data_prefix = dict(
    pcd='mmwave',
    skel='skeleton'
)

# Number of points and frames
# Since your DSP simple RDA probably outputs a different number of points, 
# 128 is a safe baseline. The point duplicator/clipper will enforce this.
point_cloud_size = 128

# Sequences of 8 frames for the dataset
num_frames = 8
backup_frames = 5
total_frames = num_frames + backup_frames

# Kinect V2 typically uses 20 or 25 joints. The custom capture defaults to 20 joints.
keypoints_involved = list(range(0, 20))

# Dataset processing pipeline
train_pipeline = [
    dict(
        type='LoadMultiFrameFromH5',
        load_pcd_dim=5,  # [x, y, z, doppler/velocity, snr]
        num_frames=num_frames,
        backup_frames=backup_frames,
        empty_frame_op='prev',
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
        tran_xyz=(0.0, 0.0, +1.685) # calculate precise tilt
        rotate_xyz=(-10.0, 0.0, 0.0),
    ),
    dict(
        type='PointCloudCoordinateTransform',
        tran_xyz=(0.0, 0.0,+0.49), 
        rotate_xyz=(21.0, 0.0, 0.0),
    ),
    
    dict(
        type='PointDuplicator',
        target_num_points=point_cloud_size
    ),
    dict(
        type='PointSortAndClip',
        target_num_points=point_cloud_size,
        sort_dim=4,  # Sort by SNR dimension
        sort_order='desc'
    ),
]

# We attach this inside a dataloader dictionary so `inspect_dataset.py` can load it
train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    shuffle=False,
    drop_last=False,
    dataset=dict(
        type='MotionDataset',
        data_root=data_root,
        info_path=f"{data_root}/{train_info}",
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        sequence_length=total_frames,
        allow_pad_sequence=False
    )
)

# Dummy configurations that tools might expect, but we don't strictly use for just visualization
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
            xyz_range=[-0.3, -0.3, -0.9, 0.3, 0.3, 1.5],
            xyz_interval=[0.3, 0.3, 0.3]
        ),
        anchor_pointnet_cfg=dict(
            channels=[24+4+3, 32, 48, 64],
            kernel_size=1
        ),
        anchor_voxelnet_cfg=dict(
            channels=[64, 96, 128, 64],
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
optimizer_cfg = dict(type='AdamW', lr=0.001)

