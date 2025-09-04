_base_ = [
    '../../../configs/__base__/default_runtime.py',
]
custom_imports = dict(
    imports=['mwpose3d', 'projects.mmmesh'], allow_failed_imports=False)

data_prefix = dict(pcd='mmwave',skel='skeleton')
data_root = './data/gaming/S15OTBPX05'
dataset_variant = 's15otbp9vly05'
dataset_variant = 'mini'
train_info = f'info_{dataset_variant}_train.pkl'
val_info = f'info_{dataset_variant}_test.pkl'
test_info = f'info_{dataset_variant}_test.pkl'
 
keypoints_involved=[0, 1, 2, 4, 5, 6, 8, 9, 10]

num_frames = 32
backup_frames = 20
stack_frames = 3
total_frames = num_frames + backup_frames
point_cloud_size = 64
y_offset = 0 # -2.0
z_offset = 1
x_tilt = -15 # 15 degrees downward
skel_x_tilt = -10
model = dict(
    type="MmMeshPredictor",
    point_cloud_size=point_cloud_size,
    frame_len=num_frames,
    criterion=[dict(type="MSELoss", weight=1.0), dict(type="SoftDTW", weight=0.1)],
    base_pointnet_cfg=dict(type="BasePointNet",channels=[6, 8, 16, 24],kernel_size=1),
    global_module_cfg=dict(type="GlobalModule",
        global_pointnet_cfg=dict(channels=[24+4, 32, 48, 64],kernel_size=1),
        global_rnn_cfg=dict(in_channel=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, 
                            fc_channels=[64, 16, 2], learnable_init_state=True, bidirectional=True)
    ),
    anchor_module_cfg=dict(type="AnchorModule", 
        anchor_cfg=dict(grouping_nsample=4,xyz_range=[-0.3, -0.3, -0.9, 0.3, 0.3, 1.5], xyz_interval=[0.3, 0.3, 0.3]),
        anchor_pointnet_cfg=dict(channels=[24+4+3, 32, 48, 64], kernel_size=1),
        anchor_voxelnet_cfg=dict(channels=[64, 96, 128, 64], kernel_size=((3,3,3), (5,1,1), (3,1,1)),),# kernel_size=((3,3,3), (5,3,3),(3,3,3),),),
        anchor_rnn_cfg=dict(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=True, learnable_init_state=True,)
    ),
    fusion_module_cfg=dict(type="SimpleKpFusionHead", channels=[256, 128, len(keypoints_involved)*3],),
)

tracking_discretize_resolution=(0.1, 0.1, 1)

TR_LoadMultiFrameFromH5 = dict(type='LoadMultiFrameFromH5', load_pcd_dim=5, num_frames=num_frames, backup_frames=backup_frames, empty_frame_op='prev')
TR_AddRangeDimension = dict(type='AddRangeDimension', insert_idx=5)
TR_StackPointCloudFrames = dict(type='StackPointCloudFrames', stack_size=stack_frames, inject_index=False, keep_structure=False)
TR_DensityFilter = dict(type='DensityFilter', mode='threshold', min_points=10, min_num_frames=num_frames)
TR_SequenceClip = dict(type='SequenceClip', mode='last', sequence_length=num_frames)
TR_SkeletonKeypointFilter = dict(type='SkeletonKeypointFilter', keypoints_involved=keypoints_involved)
TR_SkeletonCoordinateTransform = dict(type='SkeletonCoordinateTransform', rotate_xyz=(skel_x_tilt, 0, 0), tran_xyz=(0, y_offset, 0))
TR_PointCloudCoordinateTransform = dict(type='PointCloudCoordinateTransform', rotate_xyz=(x_tilt, 0, 0), tran_xyz=(0, y_offset, 0))
TR_LoadTrackingRecords = dict(type='LoadTrackingRecords', tracker_name='RKFTracker', anchor_frame='nearestperframe', centroid_queue_len=num_frames,
        ignore_axis=[2], tracker_cfg='configs/apps/trackers/rkf.py',) # point_cloud_clipping=dict(dimensions=(2,), threshold=((None, -1),)))
TR_PointDuplicator = dict(type='PointDuplicator', target_num_points=point_cloud_size)
TR_PointSortAndClip = dict(type='PointSortAndClip', target_num_points=point_cloud_size, sort_dim=4, sort_order='desc')
TR_NormalizePointAttr = dict(type='NormalizePointAttr', attr_indices=(3, 4,), means=(0.0430, 9.4746), stds=(1.3270, 9.2075))
TR_RelativeCoordtoTrackingCentroid = dict(type='RelativeCoordtoTrackingCentroid', discretize_resolution=tracking_discretize_resolution)
TR_TrackingCentroidCalibration = dict(type='TrackingCentroidCalibration', method="suppress", method_cfg=dict(
    keypoints_involved=keypoints_involved, spine_idx=1, l_shoulder_idx=4, l_wrist_idx=6, r_shoulder_idx=8, r_wrist_idx=10,))
TR_SmoothingTrackingCentroid = dict(type='SmoothingTrackingCentroid', jitter_radius=0.1, release_scale=1.5, min_jitter_len=2, still_frames=3, window_length=7, polyorder=2)

PR_SkeletonBackToOriginalCoord = dict(type='SkeletonBackToOriginalCoord')
PR_SavGolayFilter = dict(type='SavGolayFilter', window_length=7, polyorder=2, deriv=0, delta=0.055, mode='reflect', time_axis=0)

default_pipeline = [
    TR_LoadMultiFrameFromH5,
    TR_AddRangeDimension,
    TR_StackPointCloudFrames,
    TR_DensityFilter,
    TR_SequenceClip,
    TR_SkeletonKeypointFilter,
    TR_SkeletonCoordinateTransform,
    TR_PointCloudCoordinateTransform,
    TR_PointDuplicator,
    TR_PointSortAndClip,
    TR_NormalizePointAttr,
    TR_LoadTrackingRecords,
]
calib_extra_pipeline = [
    TR_TrackingCentroidCalibration
]
normal_extra_pipeline = [
    TR_SmoothingTrackingCentroid,
    TR_RelativeCoordtoTrackingCentroid
]

inference_engine_postprocess_pipeline = [
    PR_SkeletonBackToOriginalCoord
]

inference_engine_start_checkpoint = 'checkpoints/s15otbp9vly05-df10r5.pth'
load_from = inference_engine_start_checkpoint
inference_engine=dict(model=dict(model, return_sequence=True), 
            total_frames=num_frames, test_pipeline=normal_extra_pipeline, postprocess=inference_engine_postprocess_pipeline,
            load_from=inference_engine_start_checkpoint, keypoints_involved=keypoints_involved, custom_hooks=None)


TR_RandomTransform = dict(type='RandomTransform', transform_prob=0.5, sigma_xyz=(0.02, 0.02, 0.1), max_d_xyz=(0.05, 0.05, 0.4))
train_pipeline = [
    *default_pipeline,
    TR_RandomTransform
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

optimizer_cfg = dict(type='AdamW', lr=0.00025, weight_decay=0.01)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)

val_pipeline = [
    *default_pipeline,
]
inference_pipeline = [
    *default_pipeline,
    *calib_extra_pipeline,
    *normal_extra_pipeline,
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
    PR_SkeletonBackToOriginalCoord, PR_SavGolayFilter
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
            "keypoints_involved": keypoints_involved,
            "keypoint_for_stats": [5, 6],
            "error_type": "abs_error",
            "window_size": 100,
            "follow": True,
            "max_points_per_frame": 128,
        }
    ),
    postprocess=postprocess,
)

custom_hooks = [
    dict(
        type='PreInferenceHook',
        inference_engine=inference_engine,
         extra_pipeline=calib_extra_pipeline,
         earliest_activation_epoch=0,
         dynamic_loading_start_epoch=60,
         strict_loading=True,
        parallel=False,
         priority="NORMAL"
    ),
    dict(
        type='ExtraTransformHook',
        extra_pipeline=normal_extra_pipeline,
        parallel=False,
        priority="BELOW_NORMAL"
    )
]
inference_hooks = []