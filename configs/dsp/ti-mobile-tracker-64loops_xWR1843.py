# DSP config for offline point cloud reprocessing.
# Hardware: xWR1843, TI mobile-tracker pipeline, 64 loops/frame.
#
# Use with tools/create_aux_data.py --pcd to generate an alternative
# point cloud variant from raw ADC .bin files.
#
# has_timestamp: set True when the .bin file embeds an 8-byte float64
# timestamp header before each frame (i.e. recorded with DCA1000
# timestamp mode enabled).  If False, frames cannot be matched to
# the existing formatted dataset by timestamp and processing will fail.

has_timestamp = True

mmwave_radar_cfg = dict(
    mode="3D",
    num_tx=3,
    num_rx=4,
    loops_per_frame=64,
    adc_samples=256,
    start_freq_ghz=60.5,
    freq_slope_mhz_us=50.0,
    sample_rate_ksps=6000,
    idle_time_us=100,
    ramp_end_time_us=69,
    range_bias_m=0.0,
)

dsp_pipeline_cfg = [
    dict(type="FrameReshaper"),

    # Range FFT
    dict(type="RangeFFT", window="hann"),

    # DC range signature removal
    dict(
        type="CalibDcRangeSig",
        enabled=True,
        negative_bin_idx=-5,
        positive_bin_idx=8,
        num_avg_chirps=256,
        subtract_during_estimation=False,
    ),

    # Doppler FFT with clutter removal
    dict(type="DopplerFFT", window="hamming", clutter_removal=True),

    # CFAR range direction
    dict(
        type="RangeCFAR",
        averaging_mode="CASO",
        noise_win=8,
        guard_len=4,
        threshold_db=15,
        cyclic=False,
        div_shift=None,
    ),

    # CFAR Doppler direction
    dict(
        type="DopplerCFAR",
        averaging_mode="CA",
        noise_win=4,
        guard_len=2,
        threshold_db=15.0,
        cyclic=True,
        div_shift=None,
        peak_grouping=False,
        fov_range_m=(0.0, 11.11),
        fov_doppler_mps=(-2.04, 2.04),
        min_snr_db=None,
    ),

    # AoA: azimuth + elevation
    dict(
        type="AoA_TI_DPU",
        num_angle_bins=64,
        points_first=True,
        azimuth_tx_indices=(0, 1),
        elevation_tx_index=2,
        multi_obj_enable=False,
        multi_obj_thresh=0.5,
        aoa_fov_az_deg=(-90, 90),
        aoa_fov_el_deg=(-45, 45),
    ),
]
