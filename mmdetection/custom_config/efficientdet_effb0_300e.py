auto_scale_lr = dict(base_batch_size=128, enable=False)
backend_args = None
batch_augments = [
    dict(size=(
        512,
        512,
    ), type='BatchFixedSizePad'),
]
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth'
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.EfficientDet.efficientdet',
    ])
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=2, max_keep_ckpts=2, save_best='auto', type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evalute_type = 'CocoMetric'
image_size = 512
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 300
model = dict(
    backbone=dict(
        arch='b0',
        conv_cfg=dict(type='Conv2dSamePadding'),
        drop_path_rate=0.2,
        frozen_stages=0,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth',
            prefix='backbone',
            type='Pretrained'),
        norm_cfg=dict(
            eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN'),
        norm_eval=False,
        out_indices=(
            3,
            4,
            5,
        ),
        type='EfficientNet'),
    bbox_head=dict(
        anchor_generator=dict(
            center_offset=0.5,
            octave_base_scale=4,
            ratios=[
                1.0,
                0.5,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=64,
        in_channels=64,
        loss_bbox=dict(beta=0.1, loss_weight=50, type='HuberLoss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=1.5,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        norm_cfg=dict(
            eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN'),
        num_classes=10,
        num_ins=5,
        stacked_convs=3,
        type='EfficientDetSepBNHead'),
    data_preprocessor=dict(
        batch_augments=[
            dict(size=(
                512,
                512,
            ), type='BatchFixedSizePad'),
        ],
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=512,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            40,
            112,
            320,
        ],
        norm_cfg=dict(
            eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN'),
        num_stages=3,
        out_channels=64,
        start_level=0,
        type='BiFPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(
            iou_threshold=0.3,
            method='gaussian',
            min_score=0.001,
            sigma=0.5,
            type='soft_nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.5,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='EfficientDet')
norm_cfg = dict(eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    _scope_='mmdet',
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(lr=0.16, momentum=0.9, type='SGD', weight_decay=4e-05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=917, start_factor=0.1, type='LinearLR'),
    dict(
        T_max=10,
        begin=10,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=0.0,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = None
test_dataloader = None
test_evaluator = None
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    _scope_='mmdet', max_epochs=20, type='EpochBasedTrainLoop', val_interval=2)
train_dataloader = dict(
    batch_sampler=dict(_scope_='mmdet', type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        _scope_='mmdet',
        ann_file='train_1.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='../../data/dataset/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
                'General trash',
                'Paper',
                'Paper pack',
                'Metal',
                'Glass',
                'Plastic',
                'Styrofoam',
                'Plastic bag',
                'Battery',
                'Clothing',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    106,
                    0,
                    228,
                ),
                (
                    60,
                    20,
                    220,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    0,
                    0,
                    70,
                ),
                (
                    50,
                    0,
                    192,
                ),
                (
                    250,
                    170,
                    30,
                ),
                (
                    255,
                    0,
                    0,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                scale=(
                    512,
                    512,
                ),
                type='RandomResize'),
            dict(crop_size=(
                512,
                512,
            ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            512,
            512,
        ),
        type='RandomResize'),
    dict(crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(_scope_='mmdet', type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        _scope_='mmdet',
        ann_file='val_1.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='../../data/dataset/',
        metainfo=dict(
            classes=(
                'General trash',
                'Paper',
                'Paper pack',
                'Metal',
                'Glass',
                'Plastic',
                'Styrofoam',
                'Plastic bag',
                'Battery',
                'Clothing',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    230,
                ),
                (
                    106,
                    0,
                    228,
                ),
                (
                    60,
                    20,
                    220,
                ),
                (
                    0,
                    80,
                    100,
                ),
                (
                    0,
                    0,
                    70,
                ),
                (
                    50,
                    0,
                    192,
                ),
                (
                    250,
                    170,
                    30,
                ),
                (
                    255,
                    0,
                    0,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='../../data/dataset/val_1.json',
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(
                entity='cv-12',
                group='jay_dataset1',
                name='efficientdet_effb0_bifpn_8xb16-crop512-300e_coco',
                project='trash_detection'),
            type='WandbVisBackend'),
    ])
work_dir = './work_dirs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco_dataset1'