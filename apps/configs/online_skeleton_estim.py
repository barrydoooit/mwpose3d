type='infengine'
reader_cfg=dict(
    type='BufferedPcdReaderIWR6843',
    CLI_port='COM4',
    Data_port='COM5',
    config_file_path='./chirp_configs/6843_mobile_tracker.cfg'
    # config_file_path='./chirp_configs/test.cfg'
)
loop_cfg=dict(
    interval=0.1,
)
visualize = True

deep_model_cfg_path = './dl_engine/configs/mmmesh_anchorrevised.py'
checkpoint_path = './dl_engine/checkpoints/mmmesh_latest.pth'