#mmdetection 버전 3 설치 가이드
##1. 가상환경 세팅
##2. pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 설치
##3. https://mmdetection.readthedocs.io/en/latest/get_started.html 사이트 참고해서 데모 돌아가는지 확인
##4. Common_config.py 생성(mmdetection하위에 바로)
##5. submission_hook.py 훅 생성(mmdetection/mmdet/engine/hooks에)
##6. __init__.py 훅 등록 (mmdetection/mmdet/engine/hooks에)
##7. train.py생성 및 실행