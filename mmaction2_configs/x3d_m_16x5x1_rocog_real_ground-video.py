_base_ = ['../_base_/models/x3d.py', '../_base_/default_runtime.py']

model = dict(
    cls_head=dict(num_classes=7))
# dataset settings
split = 1
dataset_type_train = 'VideoDataset'
dataset_type_test = 'VideoDataset'
data_root = '/home/paulwa1/HRTII_Vision/RoCoG/uav_ground_real_cropped_video/'
ann_file_train = f'/home/paulwa1/HRTII_Vision/RoCoG/splits/umd_uav_ground_real_ground_train.txt'
ann_file_val = '/home/paulwa1/HRTII_Vision/RoCoG/splits/umd_uav_ground_real_ground_val.txt'
ann_file_test = f'/home/paulwa1/HRTII_Vision/RoCoG/splits/umd_uav_ground_real_air_test.txt'

img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.38, 57.38, 57.38], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit', num_threads=2),
    dict(type='SampleFrames', clip_len=16, frame_interval=5, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=256,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit', num_threads=2),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=24,
    workers_per_gpu=32,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type_train,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        start_index=0),
    val=dict(
        type=dataset_type_test,
        ann_file=ann_file_val,
        data_prefix=data_root,
        pipeline=val_pipeline,
        start_index=0),
    test=dict(
        type=dataset_type_test,
        ann_file=ann_file_val,
        data_prefix=data_root,
        pipeline=val_pipeline,
        start_index=0))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
# optimizer
optimizer = dict(
    type='SGD', lr=0.0025, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[20, 40])
log_config = dict(
    interval=5, hooks=[
        dict(type='TextLoggerHook'),
    ])
# learning policy
total_epochs = 20
dist_params = dict(backend='nccl')
checkpoint_config = dict(interval=5)
load_from = 'https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
work_dir = './work_dirs/x3d_m_16x5x1_rocog_real_ground-video/run1'
find_unused_parameters = False