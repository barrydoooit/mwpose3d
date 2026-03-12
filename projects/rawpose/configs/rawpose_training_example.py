_base_ = [
    './dca1000evm_default_config.py',
]
custom_imports = {
    'imports': ['mwpose3d', 'projects.mmmesh'],
    'allow_failed_imports': False
}

# ----------------------------------------------------------------------------
# DATASET CONFIGURATION
# ----------------------------------------------------------------------------
# Set this to the path where your dataset is stored.
data_root = './data/joaquin_3' 

# Define exactly which info files to use for train, val, and test splits.
train_info = 'info_train.pkl'
val_info = 'info_val.pkl'
test_info = 'info_test.pkl'

# Important baseline variables inherited from dca1000evm_default_config:
point_cloud_size = 128
num_frames = 8
backup_frames = 5
total_frames = num_frames + backup_frames
keypoints_involved = list(range(0, 20))

# data_prefix is inherited from dca1000evm_default_config.py
# If your point clouds or skeletons are named differently inside the HDF5, update this.
data_prefix = dict(pcd='mmwave.pointcloud.default', skel='skeleton')

# ----------------------------------------------------------------------------
# DATALOADERS
# ----------------------------------------------------------------------------
# The train_dataloader is inherited from dca1000evm_default_config.py.
# However, we need to override the batch size, root, and allow shuffling for actual training.
train_dataloader = dict(
    batch_size=32,       # Increase this based on your GPU vRAM (e.g. 128 or 256)
    num_workers=4,       # Increase for faster data loading
    shuffle=True,        # Must be True for training
    drop_last=True,     
    dataset=dict(
        data_root=data_root,
        info_path=f"{data_root}/{train_info}",
    )
)

# A validation dataloader is strictly required for the ValLoop to run
val_dataloader = dict(
    batch_size=1,        # Typically 1 for validation/testing to simplify metric collection
    num_workers=4,
    shuffle=False,       # No shuffling during validation
    dataset=dict(
        data_root=data_root,
        info_path=f"{data_root}/{val_info}",
    )
)

# A test dataloader is strictly required for the TestLoop to run
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    shuffle=False,
    dataset=dict(
        data_root=data_root,
        info_path=f"{data_root}/{test_info}",
    )
)

# ----------------------------------------------------------------------------
# TRAINING LOOPS AND OPTIMIZATION
# ----------------------------------------------------------------------------
# The base config only defines optimizer_cfg. 
# We redefine it to be more typical for training (e.g., adding weight decay)
optimizer_cfg = dict(type='AdamW', lr=0.001, weight_decay=0.01)

# Describe how long to train, and how often to validate
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=50, 
    val_interval=5
)

# ----------------------------------------------------------------------------
# METRICS AND EVALUATION
# ----------------------------------------------------------------------------
# Essential field for testing/evaluating. This tells the system how to score the network.
metric = dict(
    type='SimpleGTPredAnalyzer',
    keypoints_involved=keypoints_involved,
)

visualizer_cfg = {
    "keypoints_involved": keypoints_involved,
    "keypoint_for_stats": [5, 6],   # e.g., Left/Right Wrist
    "error_type": "abs_error",
    "window_size": 100,
    "follow": True,
    "max_points_per_frame": point_cloud_size,
}

val_cfg = dict(
    type='ValLoop',
    metric_cfg=dict(metric, visualizer_cfg=visualizer_cfg)
)

# Postprocessing logic to run before metric evaluation
PR_SkeletonBackToOriginalCoord = dict(type='SkeletonBackToOriginalCoord')
PR_SavGolayFilter = dict(type='SavGolayFilter', window_length=7, polyorder=2, deriv=0, delta=0.055, mode='reflect', time_axis=0)

postprocess = [
    PR_SkeletonBackToOriginalCoord, PR_SavGolayFilter
]

test_cfg = dict(
    type='TestLoop',
    metric_cfg=dict(metric, visualizer_cfg=visualizer_cfg),
    postprocess=postprocess,
)

# Optional hooks for logging, visualizations, etc.
custom_hooks = []
