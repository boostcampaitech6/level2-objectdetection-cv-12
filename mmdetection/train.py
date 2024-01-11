from mmengine.config import Config
from mmengine.runner import Runner
config = Config.fromfile('Common_Config.py')
config.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'
runner = Runner.from_cfg(config)
runner.train()
runner.test()