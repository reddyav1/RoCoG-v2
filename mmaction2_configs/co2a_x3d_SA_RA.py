# _base_ = ['../_base_/models/x3d.py', '../_base_/default_runtime.py']
_base_ = ['../../_base_/default_runtime.py']

backbone_co2a = dict(type='X3D', gamma_w=1, gamma_b=2.25, gamma_d=2.2)

in_channels = 1024 # For I3D 
in_channels = 432 # For X3D 
crop_size = 256
# pretrained_path = 'https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
# pretrained_path = '//cis/home/kshah/arl/icra_dataset/mmaction2-syn2real/work_dirs/pretrain_source/co2a_x3d_SA_bs_16_lr_2e-2_ep5_mlp_weights_pretrained/best_top1_acc_epoch_5.pth'
pretrained_path = '/cis/home/kshah/arl/icra_dataset/mmaction2-syn2real/work_dirs/pretrain_source/co2a_x3d_SA_bs_16_lr_2e-2_ep5_mlp_weights_pretrained/epoch_1.pth'

model = dict(
    type='Recognizer3D_CO2A',
    backbone=dict(
        type='CO2AVideoModel',
        backbone_co2a=backbone_co2a),
        #pretrained_path=pretrained_path),
    cls_head=dict(
        type='CO2AHead',
        num_classes=7,
        in_channels=in_channels),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
split = 1
dataset_type_train = 'VideoDataset'
dataset_type_test = 'VideoDataset'

data_root = '/cis/net/io72b/data/kshah/datasets/rocog_v2'
ann_file_train_source = '/cis/net/io72b/data/kshah/datasets/rocog_v2/annotations/syn_air_train.txt'
ann_file_train_target = '/cis/net/io72b/data/kshah/datasets/rocog_v2/annotations/real_air_train.txt'
ann_file_val = '/cis/net/io72b/data/kshah/datasets/rocog_v2/annotations/syn_air_val.txt'
ann_file_test = '/cis/net/io72b/data/kshah/datasets/rocog_v2/annotations/real_air_test.txt'

# data_root = '/cis/home/kshah/arl/icra_dataset/rocog_data/uav_ground_real_cropped_video/'
# ann_file_train_source = f'/cis/home/kshah/arl/icra_dataset/splits/umd_uav_ground_real_ground_train.txt'
# ann_file_train_target = f'/cis/home/kshah/arl/icra_dataset/splits/umd_uav_ground_real_ground_train.txt'
# ann_file_val = '/cis/home/kshah/arl/icra_dataset/splits/umd_uav_ground_real_ground_val.txt'
# ann_file_test = f'/cis/home/kshah/arl/icra_dataset/splits/umd_uav_ground_real_ground_test.txt'


img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.38, 57.38, 57.38], to_bgr=False)
train_pipeline_source = [
    dict(type='DecordInit', num_threads=2),
    # dict(type='SampleFrames', clip_len=16, frame_interval=5, num_clips=1),
    dict(type='MultiClipSampleFrames', n_frames_per_clip=16, num_clips=4),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=256,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(crop_size, crop_size), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='1NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
train_pipeline_target = [
    dict(type='DecordInit', num_threads=2),
    dict(type='MultiClipSampleFrames', n_frames_per_clip=16, num_clips=4),
    dict(type='DecordDecode', two_aug=True),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=256,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13,
        two_aug=True),
    dict(type='Resize', scale=(crop_size, crop_size), keep_ratio=False, two_aug=True),
    dict(type='Flip', flip_ratio=0.5, two_aug=True),
    dict(type='ColorJitter', two_aug=True),
    dict(type='Normalize', two_aug=True, **img_norm_cfg),
    dict(type='FormatShape', input_format='1NCTHW', two_aug=True),
    dict(type='Collect', keys=['imgs_1', 'imgs_2', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs_1', 'imgs_2', 'label'])
]
val_pipeline = [
    dict(type='DecordInit', num_threads=2),
    dict(type='MultiClipSampleFrames', n_frames_per_clip=16, num_clips=4),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=crop_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='1NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit', num_threads=2),
    dict(type='MultiClipSampleFrames', n_frames_per_clip=16, num_clips=4),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=crop_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='1NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='CO2ADataset',
        datasets=[
        dict(
            type=dataset_type_train,
            ann_file=ann_file_train_source,
            data_prefix=data_root,
            pipeline=train_pipeline_source,
            start_index=0),
        dict(
            type=dataset_type_train,
            ann_file=ann_file_train_target,
            data_prefix=data_root,
            pipeline=train_pipeline_target,
            start_index=0),
        ]),
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
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy', 'confusion_matrix']) #, gpu_collect=True)
# optimizer
optimizer = dict(
    type='SGD', lr=1e-2, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[30, 60, 90], gamma=0.5)
log_config = dict(
    interval=5, hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# learning policy
total_epochs = 50
dist_params = dict(backend='nccl')
checkpoint_config = dict(interval=5)
# load_from = 'https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
load_from = pretrained_path
work_dir = './work_dirs/scratch'
find_unused_parameters = True