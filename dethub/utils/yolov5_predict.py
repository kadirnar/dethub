from dethub.yolov5_model import Yolov5DetectionModel
from dethub.utils.data_utils import  download_yolov5n_model

def model_prediction(img, confidence_threshold, device, model_path=None):
    download_yolov5n_model()
    detection_model = Yolov5DetectionModel(confidence_threshold = 0.5, device = "cpu", model_path="dethub/yolov5/yolov5n.pt")
    prediction_list = detection_model.object_prediction_list(img)
    return prediction_list