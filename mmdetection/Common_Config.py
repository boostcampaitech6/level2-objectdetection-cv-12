# 사용하고 싶은 model의 config 파일을 불러옵니다.
_base_ = '/data/ephemeral/home/mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco.py'

# model 끝단에 num_classes 부분을 바꿔주기 위해 해당 모듈을 불러와 선언해줍니다.
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))

# custom hooks을 만들었다면 여기서 선언해 사용할 수 있습니다.
custom_hooks = [
    dict(type='SubmissionHook'),
]

# dataset 설정을 해줍니다.
data_root = '/data/ephemeral/home/dataset/'
metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img='')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img='')))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.json',
        data_prefix=dict(img='')))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'train.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    )
test_evaluator = dict(ann_file=data_root + 'test.json')

# 사용하는 모델의 pre-trainede된 checkpoint 경로/링크를 불러옵니다.
load_from = 'https://download.openxlab.org.cn/models/mmdetection/FasterR-CNN/weight/faster-rcnn_r50-caffe_fpn_1x_coco'
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)