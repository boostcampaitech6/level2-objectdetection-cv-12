from ploty_visualization import PlotlyVisualizer
from utils.path import file_path, target_image_id, image_path

if __name__ == "__main__":
    plotly_vis = PlotlyVisualizer(target_image_id, file_path, image_path)
    plotly_vis.visualization_bbox()