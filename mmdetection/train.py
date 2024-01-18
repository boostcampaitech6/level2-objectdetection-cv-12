from mmengine.config import Config
from mmengine.runner import Runner

# Load the configuration file
config = Config.fromfile('Common_Config.py')
config.work_dir = './work_dirs/dino_swin_1'
config.default_hooks.checkpoint=dict(type='CheckpointHook',save_best='coco/bbox_mAP_50', interval=5, max_keep_ckpts=3)
# Create a Runner from the configuration
runner = Runner.from_cfg(config)
# Train the model
runner.train()

# Test the model
runner.test()