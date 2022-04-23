from matplotlib import image
from dethub.utils.data_utils import  download_yolov5n_model
from dethub.yolov5_model import Yolov5DetectionModel
import cv2

download_yolov5n_model()
confidence_threshold = 0.5
device = "cpu"
model_path = "dethub/yolov5/yolov5n.pt"
img_path = "dethub/data/small-vehicles1.jpeg"
img = cv2.imread(img_path)


detection_model = Yolov5DetectionModel(model_path, confidence_threshold, device)
detection_model.load_model()
prediction_list = detection_model.object_prediction_list(img)
