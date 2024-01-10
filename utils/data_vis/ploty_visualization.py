from utils.header import cv2, go
from data_utils import get_label_class, get_label_color, get_prediction_string_csv, get_prediction_string_json

class PlotlyVisualizer:
    def __init__(self, image_name, file_path, image_path):
        self.image_name = image_name
        self.image_path = f'{image_path}/{image_name}'
        self.file_extension = file_path.split('.')[-1]
        self.file_path = file_path

    def visualize_bbox_combined_plot(self, image, bboxes):
        '''
        pascal format csv 시각화
        '''
        fig = go.Figure()

        for bbox in bboxes:
            label, score, xmin, ymin, xmax, ymax = bbox

            # 라벨에 대응하는 색상, 클래스 가져오기
            color = get_label_color(int(label))
            class_name = get_label_class(int(label))

            score = f'{score:.2f}'
            # bbox를 직사각형으로 그리기
            # legendgroup=class_name: class_name으로 group 나눠짐
            fig.add_trace(go.Scatter(
                x=[xmin, xmax, xmax, xmin, xmin],
                y=[ymin, ymin, ymax, ymax, ymin],
                mode='lines+text',
                line=dict(color=color, width=2),
                name=f'{class_name} {score}',
                text=[f'{class_name} {score}'],
                hoverinfo='text',
                textfont=dict(color='black'),
                # legendgroup=class_name, # 범례를 통해서 class_name으로 범례보는 것 가능
            ))

        fig.add_trace(go.Image(z=image))

        fig.update_layout(
            autosize=False,  # 자동 크기 조정 비활성화
            width=1300,       # 원하는 너비로 설정
            height=1300       # 원하는 높이로 설정
        )

        fig.show()

    def visualize_bbox_combined_plot_coco(self, image, bboxes):
        '''
        coco format json 시각화
        '''
        
        fig = go.Figure()

        for bbox in bboxes:
            class_id, _, x, y, width, height = bbox

            # 라벨에 대응하는 색상, 클래스 가져오기
            color = get_label_color(int(class_id))
            class_name = get_label_class(int(class_id))

            # bbox를 직사각형으로 그리기
            fig.add_trace(go.Scatter(
                x=[x, x + width, x + width, x, x],
                y=[y, y, y + height, y + height, y],
                mode='lines+text',
                line=dict(color=color, width=2),
                name=f'{class_name}',
                text=[f'{class_name}'],
                hoverinfo='text',
                textfont=dict(color='black'),
            ))

        fig.add_trace(go.Image(z=image))

        fig.update_layout(
            autosize=False,  # 자동 크기 조정 비활성화
            width=1300,       # 원하는 너비로 설정
            height=1300       # 원하는 높이로 설정
        )

        fig.show()

    def visualization_bbox(self):
        if self.file_extension == 'csv':
            bboxes = get_prediction_string_csv(self.file_path, self.image_name)

            if not bboxes:
                print("Empty prediction_string.")
                return
        
            print(bboxes)
            image = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
            # bboxes = self.parse_prediction_string_to_bboxes(prediction_string)

            self.visualize_bbox_combined_plot(image, bboxes)

        elif self.file_extension == 'json':
            bboxes = get_prediction_string_json(self.file_path, self.image_name)

            if not bboxes:
                print("Empty prediction_string.")
                return
        
            print(bboxes)
            image = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
            self.visualize_bbox_combined_plot_coco(image, bboxes)
        
        else:
            print('This is not a CSV or JSON file.')