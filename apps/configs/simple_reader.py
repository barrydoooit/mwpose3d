type='simple'
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