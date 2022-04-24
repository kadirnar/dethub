def yolov5_predict(img, model_path, device, confidence_threshold):
    from dethub.utils.vis import yolov5_visualization
    from dethub.utils.model_predict import yolov5_prediction
    from dethub.utils.data_utils import read_image

    img = read_image(img)
    prediction_list = yolov5_prediction(img, confidence_threshold, device, model_path)
    img = yolov5_visualization(img, prediction_list)


def torchvision_predict(img, model_path, confidence_threshold):
    from dethub.utils.vis import torchvision_visualization
    from dethub.utils.model_predict import torchvision_prediction
    from dethub.utils.data_utils import read_image

    img = read_image(img)
    img, prediction_boxes, prediction_class, prediction_score = torchvision_prediction(img, model_path, confidence_threshold)
    img = torchvision_visualization(img, prediction_boxes, prediction_score, prediction_class)
