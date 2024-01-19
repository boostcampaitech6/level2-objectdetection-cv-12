# 사용하고 싶은 model의 config 파일을 불러옵니다.
# _base_ = '/data/ephemeral/home/mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco.py'
# _base_ = '/data/ephemeral/home/mmdetection/configs/cascade_rcnn/cascade-rcnn_r50-caffe_fpn_1x_coco.py'
# _base_ = '/data/ephemeral/home/mmdetection/configs/cascade_rcnn/cascade-rcnn_x101-64x4d_fpn_1x_coco.py'
# _base_ = '/data/ephemeral/home/mmdetection/configs/dino/dino-4scale_r50_8xb2-24e_coco.py'
# _base_ = '/data/ephemeral/home/mmdetection/configs/dino/dino-4scale_r50_8xb2-24e_coco.py'
# _base_ = '/data/ephemeral/home/mmdetection/configs/ddq/ddq-detr-4scale_swinl_8xb2-30e_coco.py'
_base_ = '/data/ephemeral/home/mmdetection/configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py'

# model 끝단에 num_classes 부분을 바꿔주기 위해 해당 모듈을 불러와 선언해줍니다.
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(
#             type='Shared2FCBBoxHead',
#             in_channels=256,
#             fc_out_channels=1024,
#             roi_feat_size=7,
#             num_classes=10,
#             bbox_coder=dict(
#                 type='DeltaXYWHBBoxCoder',
#                 target_means=[0., 0., 0., 0.],
#                 target_stds=[0.1, 0.1, 0.2, 0.2]),
#             reg_class_agnostic=False,
#             loss_cls=dict(
#                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#             loss_bbox=dict(type='L1Loss', loss_weight=1.0))))
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
model = dict(bbox_head=dict(
        type='DINOHead',
        num_classes=10,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0))    
        )
# roi_head=dict(
#         type='CascadeRoIHead',
#         num_stages=3,
#         stage_loss_weights=[1, 0.5, 0.25],
#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         bbox_head=[
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=10,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.1, 0.1, 0.2, 0.2]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=10,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.05, 0.05, 0.1, 0.1]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=10,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.033, 0.033, 0.067, 0.067]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
#         ])
# model = dict(
#     bbox_head=dict(
#         type='DDQDETRHead',
#         num_classes=10,
#         sync_cls_avg_factor=True,
#         loss_cls=dict(
#             type='FocalLoss',
#             use_sigmoid=True,
#             gamma=2.0,
#             alpha=0.25,
#             loss_weight=1.0),
#         loss_bbox=dict(type='L1Loss', loss_weight=5.0),
#         loss_iou=dict(type='GIoULoss', loss_weight=2.0))
# )
# custom hooks을 만들었다면 여기서 선언해 사용할 수 있습니다.
custom_hooks = [
    dict(type='SubmissionHook'),
]

# dataset 설정을 해줍니다.
data_root = '/data/ephemeral/home/Git/utils/valid_set/kfold/'
metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_1.json',
        data_prefix=dict(img='')))

valid_name = 'val_1.json'

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=valid_name,
        data_prefix=dict(img='')))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.json',
        data_prefix=dict(img='')))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + valid_name,
    metric='bbox',
    format_only=False,
    classwise=True,
    )
test_evaluator = dict(ann_file=data_root + 'test.json')

# 사용하는 모델의 pre-trainede된 checkpoint 경로/링크를 불러옵니다.
# load_from = 'https://download.openxlab.org.cn/models/mmdetection/FasterR-CNN/weight/faster-rcnn_r50-caffe_fpn_1x_coco'
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco/cascade_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.404_20200504_174853-b857be87.pth'
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-24e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_swinl_30e.pth'
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq-detr-4scale_r50_8xb2-12e_coco/ddq-detr-4scale_r50_8xb2-12e_coco_20230809_170711-42528127.pth'
# load_from = '/data/ephemeral/home/mmdetection/work_dirs/ddq-detr-4scale_r50_8xb2-12e_coco/best_coco_bbox_mAP_50_epoch_12.pth'
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)


dataset_no = 1
# max_epochs = 12
# cfg_filename = 'cascade-rcnn_r50-caffe_fpn_1x_coco'
# cfg_filename = 'cascade-rcnn_x101-64x4d_fpn_1x_coco'
# cfg_filename = 'dino-4scale_r50_8xb2-12e_coco'
cfg_filename = 'dino-5scale_swin-l_8xb2-12e_coco'
# cfg_filename = 'ddq-detr-4scale_swinl_8xb2-30e_coco'
# cfg_filename = 'ddq-detr-4scale_r50_8xb2-12e_coco'
vis_backends = [dict(type='LocalVisBackend'),
                    dict(type='WandbVisBackend', 
                         init_kwargs={
                             'entity': 'cv-12',
                             'project': 'trash_detection',
                             'group': f'dataset{dataset_no}',
                             'name': cfg_filename
                             })]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')