reader_cfg=dict(
    type='BufferedPcdReaderIWR6843',
    
    # === Serial Port Configuration ===
    # Windows: COM3, COM4, etc.
    # Linux: /dev/ttyUSB0, /dev/ttyUSB1, etc.
    # macOS: /dev/tty.usbmodemXXXX (run `ls /dev/tty.usbmodem*` to find)
    #
    # IWR6843 has two USB ports:
    #   - CLI_port: For sending configuration commands (usually "Enhanced COM Port")
    #   - Data_port: For receiving point cloud data (usually "User COM Port")
    
    # macOS ports (uncomment and update when radar is connected)
    CLI_port='/dev/tty.SLAB_USBtoUART',
    Data_port='/dev/tty.SLAB_USBtoUART3',
    
    # Windows ports (default)
    # CLI_port='COM4',
    # Data_port='COM3',
    
    # Linux ports
    # CLI_port='/dev/ttyUSB0',
    # Data_port='/dev/ttyUSB1',
    
    density_threshold=3,
    config_file_path='./configs/chirp_configs/6843_tracking.cfg',
    firmware_tilt_deg=20,
    firmware_tilt_axis='x',
    undo_firmware_tilt=True,
    point_cloud_range=[-1.75, 0.5, -1.5, 1.75, 4.5, 2],
)
