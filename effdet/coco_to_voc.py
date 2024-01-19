import json
import csv

def coco_to_voc(coco_json_file, output_dir):
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)

    voc = {}
    for annotation in coco_data:
        id = 'test/{0:04d}.jpg'.format(annotation['image_id'])
        if id not in voc.keys():
            voc[id] = ''
        voc[id] += str(annotation['category_id']) + ' '
        voc[id] += str(annotation['score']) + ' '
        voc[id] += str(annotation['bbox'][0]) + ' '
        voc[id] += str(annotation['bbox'][1]) + ' '
        if annotation['bbox'][0] + annotation['bbox'][2] > 1024:
            voc[id] += '1024 '
        else:
            voc[id] += str(annotation['bbox'][0] + annotation['bbox'][2]) + ' '
        if annotation['bbox'][1] + annotation['bbox'][3] > 1024:
            voc[id] += '1024 '
        else:
            voc[id] += str(annotation['bbox'][1] + annotation['bbox'][3]) + ' '


    data = [
        ['PredictionString', 'image_id']
    ]
    for image_id, PredictionString in voc.items():
        data.append([PredictionString, image_id])

    # CSV 파일 쓰기
    with open(output_dir, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 데이터 쓰기
        writer.writerows(data)


path = 'path'
coco_json_file = path + '/results.json'
output_directory = path + '/submission.csv'
coco_to_voc(coco_json_file, output_directory)
