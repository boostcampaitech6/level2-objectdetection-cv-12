# mmdetection 버전 3 설치 가이드

## 1. 가상환경 세팅
    conda create --name openmmlab python=3.8 -y
    conda activate openmmlab

## 2. torch 설치
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
## 3. 공식 사이트 참고해서 데모 돌아가는지 확인
    https://mmdetection.readthedocs.io/en/latest/get_started.html
## 4. Common_config.py 생성(mmdetection하위에 바로)
#### __base__ 부분만 경로 수정해주시면됩니다.
## 5. submission_hook.py 훅 생성(mmdetection/mmdet/engine/hooks에)
#### 이 부분 수정해주시면 됩니다.
    self.file_names.append(output.img_path[-13:])
## 6. __init__.py 훅 등록 (mmdetection/mmdet/engine/hooks에)
    from .submission_hook import SubmissionHook
    
    __all__ = [
    'YOLOXModeSwitchHook', 'SyncNormHook', 'CheckInvalidLossHook',
    'SetEpochInfoHook', 'MemoryProfilerHook', 'DetVisualizationHook',
    'NumClassCheckHook', 'MeanTeacherHook', 'trigger_visualization_hook',
    'PipelineSwitchHook', 'TrackVisualizationHook',
    'GroundingVisualizationHook', 'SubmissionHook']
## 7. train.py생성 및 실행
    from mmengine.config import Config
    from mmengine.runner import Runner
    config = Config.fromfile('Common_Config.py')
    config.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'
    runner = Runner.from_cfg(config)
    runner.train()
    runner.test()