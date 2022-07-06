from dethub.model import *


def get_prediction(image, detection_model):
    import cv2

    image = cv2.imread(image)
    prediction = detection_model.object_prediction_list(image)

    return prediction


img = "data/highway1.jpg"
detection_model = Yolov5(model_path="dethub/models/yolov5/yolov5n.pt", device="cpu", confidence_threshold=0.5)
output = get_prediction(img, detection_model)
