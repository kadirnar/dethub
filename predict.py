from dethub.utils.vis import yolov5_visualization
from dethub.utils.yolov5_predict import model_prediction
import cv2


img = cv2.imread("dethub/data/small-vehicles1.jpeg") # img path
prediction_list =  model_prediction(img, 0.5, "cpu", "dethub/yolov5/yolov5n.pt") # model prediction
img = yolov5_visualization(img, prediction_list)     # visualization

