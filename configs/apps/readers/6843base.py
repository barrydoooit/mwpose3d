reader_cfg=dict(
    type='BufferedPcdReaderIWR6843',
    CLI_port='COM4',
    Data_port='COM3',
    # CLI_port='/dev/ttyUSB0',
    # Data_port='/dev/ttyUSB1',
    config_file_path='./configs/chirp_configs/6843_tracking.cfg',
    firmware_tilt_deg=20,
    firmware_tilt_axis='x',
    undo_firmware_tilt=True,
    point_cloud_range=[-2, 2, -1.5, 2, 5, 1.5]
    # config_file_path='./configs/chirp_configs/6843_vitalsign.cfg'
    # config_file_path='./configs/chirp_configs/6843_mobile_tracker.cfg',
    # config_file_path='./configs/chirp_configs/test.cfg'
)