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

# effdet 가이드

## train
train.py를 실행하여 학습합니다.
config.yaml을 사용하면 편하게 parser들을 설정할 수 있습니다.
wandb 이름은 train.py 파일 내에서 지정해 줍니다.

## inference
validate.py를 실행하여 추론합니다.
학습시 .pth.tar 확장자로 저장된 파일에서 .tar를 지워주어야 합니다.
따로 config 파일이 없기에 parser에 적잘한 값들을 넣어주셔야 합니다.
결과값은 coco형식의 annotation 값들이 나오게 됩니다.
이 파일을 coco_to_voc.py를 통해 변경하면 제출형식에 맞는 csv 파일을 저장하게됩니다.