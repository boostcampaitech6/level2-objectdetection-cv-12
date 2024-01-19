_base_ = '/data/ephemeral/home/Git/mmdetection/configs/_base_/datasets/coco_detection.py'

train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        bbox_clip_border=True,
        center_ratio_range=(
            0.5,
            1.5,
        ),
        img_scale=(
            1024,
            1024,
        ),
        pad_val=114.0,
        prob=1.0,
        type='Mosaic'),
    dict(type='PackDetInputs'),
]