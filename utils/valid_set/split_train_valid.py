import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from collections import Counter
import pandas as pd
import os

annotation = 'dataset/train.json'

with open(annotation) as f:
    data = json.load(f)

var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
X = np.ones((len(data['annotations']),1))
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# for train_idx, val_idx in cv.split(X, y, groups):
#     print("TRAIN:", groups[train_idx])
#     print(" ", y[train_idx])
#     print(" TEST:", groups[val_idx])
#     print(" ", y[val_idx])

# 디렉토리가 없으면 생성
os.makedirs('dataset/kfold/', exist_ok=True)
os.makedirs('dataset/random/', exist_ok=True)

# JSON 파일로 저장 - KFold
for i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
    train_data = [data['annotations'][idx] for idx in train_idx]
    val_data = [data['annotations'][idx] for idx in val_idx]

    train_json = {
        "info": data['info'],
        "licenses": data['licenses'],
        "categories": data['categories'],
        "images": [img for img in data['images'] if img['id'] in groups[train_idx]],
        "annotations": train_data
    }

    val_json = {
        "info": data['info'],
        "licenses": data['licenses'],
        "categories": data['categories'],
        "images": [img for img in data['images'] if img['id'] in groups[val_idx]],
        "annotations": val_data
    }

    with open(f'dataset/kfold/train_{i}.json', 'w') as train_file:
        json.dump(train_json, train_file)

    with open(f'dataset/kfold/val_{i}.json', 'w') as val_file:
        json.dump(val_json, val_file)

X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(X, y, test_size=0.2, random_state=42)

# JSON 파일로 저장 - 랜덤 분할
random_train_json = {
    "info": data['info'],
    "licenses": data['licenses'],
    "categories": data['categories'],
    "images": [img for img in data['images'] if img['id'] in X_train_random[:, 0]],
    "annotations": [data['annotations'][idx] for idx in range(len(data['annotations'])) if idx in X_train_random[:, 0]]
}

random_val_json = {
    "info": data['info'],
    "licenses": data['licenses'],
    "categories": data['categories'],
    "images": [img for img in data['images'] if img['id'] in X_test_random[:, 0]],
    "annotations": [data['annotations'][idx] for idx in range(len(data['annotations'])) if idx in X_test_random[:, 0]]
}

with open('dataset/random/train_random.json', 'w') as random_train_file:
    json.dump(random_train_json, random_train_file)

with open('dataset/random/val_random.json', 'w') as random_val_file:
    json.dump(random_val_json, random_val_file)

# check distribution (체크가 필요없다면 주석)
def get_distribution(y):
    y_distr = Counter(y)
    y_vals_sum = sum(y_distr.values())

    return [f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) +1)]

distrs = [get_distribution(y)]
index = ['training set']

# KFold에 대한 분포도 추가
for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X,y, groups)):
    train_y, val_y = y[train_idx], y[val_idx]
    train_gr, val_gr = groups[train_idx], groups[val_idx]

    assert len(set(train_gr) & set(val_gr)) == 0
    
    distrs.append(get_distribution(train_y))
    distrs.append(get_distribution(val_y))
    index.append(f'train - fold{fold_ind}')
    index.append(f'val - fold{fold_ind}')

# Random Split에 대한 분포도 추가
distrs.append(get_distribution(y_train_random))
distrs.append(get_distribution(y_test_random))
index.append('train - random split')
index.append('val - random split')

categories = [d['name'] for d in data['categories']]
df = pd.DataFrame(distrs, index=index, columns = [categories[i] for i in range(np.max(y) + 1)])
print(df)