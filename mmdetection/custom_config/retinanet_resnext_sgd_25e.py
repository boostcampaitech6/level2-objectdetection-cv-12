auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
custom_hooks = [
    dict(type='SubmissionHook'),
]
data_root = '/data/ephemeral/home/dataset/kfold/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
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
    ])
model = dict(
    backbone=dict(
        base_width=4,
        depth=101,
        frozen_stages=1,
        groups=64,
        init_cfg=dict(
            checkpoint='open-mmlab://resnext101_64x4d', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNeXt'),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
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
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=10,
        stacked_convs=4,
        type='RetinaHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='RetinaNet')
optim_wrapper = dict(
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            16,
            22,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='/data/ephemeral/home/dataset/kfold/test.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='/data/ephemeral/home/dataset/kfold/',
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
                1333,
                800,
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
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/data/ephemeral/home/dataset/kfold/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
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
train_cfg = dict(max_epochs=40, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='/data/ephemeral/home/dataset/kfold/train_2.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='/data/ephemeral/home/dataset/kfold/',
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
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='/data/ephemeral/home/dataset/kfold/val_2.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='/data/ephemeral/home/dataset/kfold/',
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
                1333,
                800,
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
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/data/ephemeral/home/dataset/kfold/val_2.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(
                entity='cv-12',
                group='dataset2',
                name='retinanet_x101-64x4d_fpn_2x_coco',
                project='trash_detection'),
            type='WandbVisBackend'),
    ])
work_dir = '/data/ephemeral/home/mmdetection/work_dirs/retinanet_x101-64x4d_fpn_2x_coco_version2_kfold2'
