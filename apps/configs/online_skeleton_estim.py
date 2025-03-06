type='infengine'
reader_cfg=dict(
    type='BufferedPcdReaderIWR6843',
    CLI_port='COM10',
    Data_port='COM11',
    config_file_path='./chirp_configs/6843_mobile_tracker.cfg'
    # config_file_path='./chirp_configs/test.cfg'
)
loop_cfg=dict(
    interval=0.03,
)
visualize = True

deep_model_cfg_path = './dl-engine/configs/mmmesh_anchorrevised.py'
checkpoint_path = './dl-engine/checkpoints/mmmesh_latest.pth'