from utils.header import pd, json

def get_prediction_string_csv(csv_file_path, image_name):
    '''
    pascal format csv 파일에서 원하는 이미지의 prediction string 불러오는 함수
    '''
    df = pd.read_csv(csv_file_path)

    try:
        target_row = df[df['image_id'] == image_name].iloc[0]
        prediction_string = target_row['PredictionString']

        bbox_info = list(map(float, filter(None, prediction_string.split(' '))))

        bboxes = [(int(bbox_info[i]), bbox_info[i+1], bbox_info[i+2], bbox_info[i+3], bbox_info[i+4], bbox_info[i+5])
                for i in range(0, len(bbox_info), 6)]
        
        return bboxes
    
    except IndexError:
        print(f"No data found for image_id {image_name}")
        return None

def get_prediction_string_json(json_file_path, image_name):
    '''
    coco format json 파일에서 원하는 이미지의 prediction string 불러오는 함수
    '''
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    image_id = None

    for img in data['images']:
        if img['file_name'] == image_name:
            image_id = img['id']
            break
        
    if image_id is None:
        print(f"No data found for image_id {image_name}")
        return None, None

    annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

    bboxes_and_category = []

    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']
        
        # train json이라 score는 임의로 0.0
        bboxes_and_category.append((category_id, 0.0, bbox[0], bbox[1], bbox[2], bbox[3]))

    return bboxes_and_category
    
def get_label_color(label):
    label_colors = {
        0: 'red',
        1: 'black',
        2: 'blue',
        3: 'yellow',
        4: 'magenta',
        5: 'cyan',
        6: 'darkred',
        7: 'darkgreen',
        8: 'darkblue',
        9: 'brown'
    }

    return label_colors.get(label, 'white')

def get_label_class(label):
    label_class = {
        0: 'General trash',
        1: 'Paper',
        2: 'Paper pack',
        3: 'Metal',
        4: 'Glass',
        5: 'Plastic',
        6: 'Styrofoam',
        7: 'Plastic bag',
        8: 'Battery',
        9: 'Clothing'
    }

    return label_class.get(label, '?')