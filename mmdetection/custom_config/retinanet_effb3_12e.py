auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None
batch_augments = [
    dict(size=(
        896,
        896,
    ), type='BatchFixedSizePad'),
]
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=4, max_keep_ckpts=2, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_size = (
    896,
    896,
)
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 12
model = dict(
    backbone=dict(
        arch='b3',
        drop_path_rate=0.2,
        frozen_stages=5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth',
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
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=10,
        num_ins=5,
        stacked_convs=4,
        type='RetinaSepBNHead'),
    data_preprocessor=dict(
        batch_augments=[
            dict(size=(
                896,
                896,
            ), type='BatchFixedSizePad'),
        ],
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
            48,
            136,
            384,
        ],
        no_norm_on_lateral=True,
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=0,
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
            neg_iou_thr=0.5,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='RetinaNet')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.001,
        type='AdamW',
        weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=0.1, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = None
test_dataloader = None
test_evaluator = None
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        896,
        896,
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
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=4)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
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
                    0.8,
                    1.2,
                ),
                scale=(
                    896,
                    896,
                ),
                type='RandomResize'),
            dict(crop_size=(
                896,
                896,
            ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.8,
            1.2,
        ),
        scale=(
            896,
            896,
        ),
        type='RandomResize'),
    dict(crop_size=(
        896,
        896,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
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
                896,
                896,
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
    ann_file='../../data/dataset/val_1.json',
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
                entity='i-will-show-you',
                group='retinanet_effb3_fpn_8xb4-crop896-1x_coco',
                name='dataset1',
                project='test_project'),
            type='WandbVisBackend'),
    ])
work_dir = './work_dirs/retinanet_effb3_fpn_8xb4-crop896-1x_coco_dataset_1'
