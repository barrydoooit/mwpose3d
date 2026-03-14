_base_ = [
        '../../../mwCore/configs/dsp/mmmesh_xWR1843.py', # simple top 128 energy
]
# mmmesh_xWR1843.py
custom_imports = dict(
    imports=['mwpose3d', 'apps.impl.dataset_collectionv2'],
    allow_failed_imports=False,
)

type = 'HPEDatasetCollectionAppV2'

# Required by BaseMWApp: a worker factory registered in THREADS that
# returns `(worker, qthread)`.
thread_cfg = dict(
    type='RawAdcUdpFlowReaderWorker',
    reader=dict(
        type='RawAdcUdpFlowReader',
        radar_cfg=_base_.mmwave_radar_cfg,
        pipeline=dict(
            type='DspPipeline',
            radar_config=_base_.mmwave_radar_cfg,
            pipeline_cfg=_base_.dsp_pipeline_cfg,
        ),
        static_ip='192.168.33.30',
        adc_ip='192.168.33.180',
        data_port=4098,
        config_port=4096,
        buffer_size=1500,
        return_as_radar_frame=True,
    ),
)

vis_cfg = dict(
    type='OnlineSkeletonVisualizer',
    joint_cnxn=((0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7),
                (2, 8), (8, 9), (9, 10), (10, 11),
                (2, 12), (12, 13), (13, 14), (14, 15),
                (2, 16), (16, 17), (17, 18), (18, 19)),
    joint_indices=list(range(20))
)

instructions = dict(
    on_init=[
        *[dict(content=f"Capture Starts in {X} seconds.", duration=1, repeats=1) for X in range(5, 0, -1)]
    ],
    on_start=[
        dict(content="Capture in progress...", duration=1, repeats=1)
    ],
    on_stop=[
        *[dict(content=f"Capture Stopped. Waiting for next capture to start ({X}s)", duration=1, repeats=1)
          for X in range(10, 5, -1)]
    ]
)

# data_root = 'apps/impl/dataset_collectionv2/traces'
data_root = 'apps/impl/dataset_collectionv2/traces/joaquin_simple_point'
buffer_cfg = dict(
    dump_dir=f'{data_root}/raw',
    buffer_size=500,
    storage_format='raw_bin_and_pointcloud_json',
    save_with_timestamp=True,
)

kinect_cfg = dict(
    kinect_mgr_cfg=dict(
        exe_path='apps/impl/dataset_collection/DumpKinectSkeleton/bin/Release/DumpKinectSkeleton.exe',
        output_dir=f'{data_root}/kinect',
        mode=['capture', 'control'],
    )
)

# Required by HPEDatasetCollectionAppV2 state machine:
# dynamic connect/disconnect during capture.
loop_cfg = dict(
    capture_signal='reader.frame_signal',
    capture_slot='pcd_buffering_worker.enqueue_frame',
    capture_connection_type='QueuedConnection',
)

# Signal wiring for BaseMWApp. The app parser accepts both:
# - explicit signal/slot
# - shorthand on/to with aliases such as reader.array_data and popup.accepted
connections = [
    dict(on='popup.accepted', to='instruction_worker.instructionsOnInit.emit'),
    dict(on='popup.rejected', to='instruction_worker.instructionsOnInit.emit'),
    dict(on='popup.submitted', to='pcd_buffering_worker.recordMeta.emit'),
    dict(on='reader.array_data', to='visualizer.on_new_cloud', type='QueuedConnection'),
    dict(on='pcd_buffering_worker.frameCount', to='controller.on_frame_count'),
    dict(on='kinect_mgr_worker.recentSkeletonJointCoordSignal', to='visualizer.update_skeleton', type='QueuedConnection'),
    dict(on='instruction_worker.finishedOnInit', to='controller.on_init_stage_complete', type='QueuedConnection'),
    dict(on='pcd_buffering_worker.bufferFull', to='controller.on_buffer_full'),
    dict(on='pcd_buffering_worker.bufferDumped', to='kinect_mgr_worker.dumpSkeletonsSignal.emit'),
    dict(on='instruction_worker.finishedOnStop', to='controller.on_cycle_complete', type='QueuedConnection'),
]
