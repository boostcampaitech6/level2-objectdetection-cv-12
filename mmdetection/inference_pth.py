import argparse
from mmengine.config import Config
from mmengine.runner import Runner


def inference(cfg_filename):
    if cfg_filename.endswith('.py'):
        cfg_filename = cfg_filename[:-3]

    # config file 들고오기
    cfg = Config.fromfile(f'./{cfg_filename}.py')
    metainfo = {
        'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 
                    'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
        'palette': [
            (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
            (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
        ]
    }

    data_root='/data/ephemeral/home/dataset/'

    cfg.test_dataloader.dataset.metainfo = metainfo
    cfg.test_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.ann_file = f'test.json'
    cfg.test_dataloader.dataset.data_prefix = dict(img='')
    cfg.test_dataloader.batch_size = 1
    cfg.test_evaluator = dict(
        type='CocoMetric',
        ann_file=data_root + 'test.json',
        metric='bbox',
        format_only=False,
        backend_args=None)

    cfg.train_dataloader = None
    cfg.train_evaluator = None
    cfg.train_cfg = None
    cfg.optim_wrapper= None
    cfg.param_scheduler = None

    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.val_cfg = None
    # cfg.test_dataloader = None
    # cfg.test_evaluator = None
    # cfg.test_cfg = None
    num = 1
    cfg.work_dir = f'./work_dirs/dino_swin{num}_test'
    
    cfg.load_from = '/data/ephemeral/home/Git/mmdetection/work_dirs/dino_swin_1/best_coco_bbox_mAP_50_epoch_4.pth'
    runner = Runner.from_cfg(cfg)
    runner.test()


def main():
    parser = argparse.ArgumentParser(description='rtmdet_swin 을 이용해서 추론하는 코드')
    # 문자열 입력 인자
    parser.add_argument('--config', type=str, required=True, help='설정파일 이름을 입력하세요.')

    args = parser.parse_args()

    inference(args.config)

if __name__ == '__main__':
    main()