from dethub.model import *


def get_prediction(image, detection_model):
    import cv2

    from dethub.utils.visualize import vis

    image = cv2.imread(image)
    object_prediction_list = detection_model.object_prediction_list(image)
    vis(image, object_prediction_list)


img = "data/highway1.jpg"
detection_model = Yolov5(model_path="dethub/models/yolov5/yolov5n.pt", device="cpu", confidence_threshold=0.5)
get_prediction(img, detection_model)
