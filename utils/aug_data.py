import json
import albumentations as A
import cv2
import os

train_json = 'train_2' # 증강할 json 파일명

# COCO JSON 파일 경로
json_path = f'/data/ephemeral/home/dataset/{train_json}.json'

folder_path = f'dataset/{train_json}_aug'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"폴더가 생성되었습니다: {folder_path}")
else:
    print(f"이미 존재하는 폴더입니다: {folder_path}")

# JSON 파일 열기
with open(json_path, 'r') as f:
    coco_data = json.load(f)

# 이미지 정보와 bounding box 정보 추출
images = coco_data['images']
annotations = coco_data['annotations']

# 새로운 JSON 데이터 생성
new_coco_data = {
    # 기존 정보는 유지
    "info": coco_data["info"],
    "licenses": coco_data["licenses"],
    "categories": coco_data["categories"],
    "images": coco_data['images'],
    "annotations": coco_data['annotations'],
}
new_image_id = 5000
new_annotation_id = 25000

transform = A.Compose([
    # 이미지를 무작위로 회전
    A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    
    # 90도, 180도, 270도 회전
    A.OneOf([
        A.Transpose(p=0.5),
        A.Rotate(limit=180, p=0.5),
        A.Rotate(limit=90, p=0.5)
    ], p=1.0),
    
    # 무작위로 밝기와 대비 조정
    A.RandomBrightnessContrast(p=0.5),

    # 채도, 색조, 명도 변화
    A.HueSaturationValue(p=0.5),
    
    # 수평으로 뒤집기
    A.HorizontalFlip(p=0.5),
    
    # 이미지를 랜덤으로 자르기
    A.RandomCrop(height=800, width=800, p=0.8),

    # 최종 이미지 크기를 1024x1024로 조절
    A.Resize(height=1024, width=1024),
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'], min_visibility=0.2))


# 이미지 ID를 기반으로 해당 이미지의 bounding box 정보 찾기
image_ids = set(ann['image_id'] for ann in annotations)
for image_id in image_ids:
    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

    # 이미지 ID에 해당하는 이미지 정보 찾기
    image_info = next((img for img in images if img['id'] == image_id), None)

    # 이미지 파일 경로와 bounding box 정보 출력
    if image_info and image_annotations:
        image_file_name = image_info['file_name']
        image = cv2.imread("dataset/" + image_file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []
        for idx, ann in enumerate(image_annotations):
            bboxes.append(ann['bbox'])
            class_labels.append(ann['category_id'])
    else:
        print("해당 이미지에 대한 정보를 찾을 수 없습니다.")

    for c, n in ([(0, 3), (8, 2), (9, 1), (3, 1), (1, 1), (5, 1)]): # (c, n) : c번 클래스가 있는 이미지는 n번 이미지 증강
        if c in class_labels and 7 not in class_labels: # 7(플라스틱 백)이 있는 이미지는 이미지 증강하지 않음 (학습시간 증가 방지)
            for i in range(n):
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_class_labels = transformed['class_labels']

                transformed_image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

                # # transformed된 이미지와 바운딩 박스 그리기
                # for bbox in transformed_bboxes:
                #     xmin, ymin, width, height = map(int, bbox)
                #     xmax = xmin + width
                #     ymax = ymin + height
                #     cv2.rectangle(transformed_image_rgb, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

                # 저장할 이미지 파일 경로
                output_image_path = f'dataset/{train_json}_aug/{image_id}_{c}_{i}.jpg'

                # 이미지 저장
                cv2.imwrite(output_image_path, transformed_image_rgb)
                print(f"이미지가 {output_image_path}에 저장되었습니다.")

                image_data = {
                    "width": 1024,
                    "height": 1024,
                    "file_name": f'{train_json}_aug/{image_id}_{c}_{i}.jpg',
                    "license": 0,
                    "flickr_url": None,
                    "coco_url": None,
                    "date_captured": "2023-01-15 00:00:00",
                    "id": new_image_id
                }
                new_coco_data['images'].append(image_data)

                for i in range(len(transformed_class_labels)):
                    annotation_data = {
                        "image_id": new_image_id,
                        "category_id": transformed_class_labels[i],
                        "area": transformed_bboxes[i][2] * transformed_bboxes[i][3],
                        "bbox": transformed_bboxes[i],
                        "iscrowd": 0,
                        "id": new_annotation_id
                    }
                    new_coco_data['annotations'].append(annotation_data)

                    new_annotation_id += 1
                new_image_id += 1

# 나머지 코드 다음에 추가하여 JSON으로 저장
                
print(new_image_id - 5000, '장의 이미지가 생겼답니다.')
new_json_path = f'dataset/{train_json}_augmented.json'
with open(new_json_path, 'w') as new_f:
    json.dump(new_coco_data, new_f)
