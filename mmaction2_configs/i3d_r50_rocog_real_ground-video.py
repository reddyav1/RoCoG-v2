_base_ = ['../_base_/models/i3d_r50.py', '../_base_/default_runtime.py']

model = dict(
    cls_head=dict(num_classes=7))
# dataset settings
split = 1
dataset_type_train = 'VideoDataset'
dataset_type_test = 'VideoDataset'
base_dir = '/cis/net/io72b/data/kshah/datasets/rocog_v2'
data_root = base_dir
ann_file_train = f'{base_dir}/annotations/real_ground_train_tsonly.txt'
ann_file_val = f'{base_dir}/annotations/real_ground_val_tsonly.txt'
ann_file_test = f'{base_dir}/annotations/real_air_test.txt'

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
    dict(type='Flip', flip_ratio=0.5),    
    dict(type='Imgaug', transforms='default'),
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
data = dict(
    videos_per_gpu=6,
    workers_per_gpu=2,
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
        ann_file=ann_file_test,
        data_prefix=data_root,
        pipeline=test_pipeline,
        start_index=0))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix'])
# optimizer
optimizer = dict(
    type='SGD', lr=0.0025, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[40])
# lr_config = dict(
#         policy='CosineAnnealing',
#         min_lr=0)
log_config = dict(
    interval=5, hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')])

# learning policy
total_epochs = 20
dist_params = dict(backend='nccl')
checkpoint_config = dict(interval=1)
load_from = 'https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth'
work_dir = './work_dirs/x3d_rocogv2_Gr2Ar_lr5e-3/run1'
find_unused_parameters = False