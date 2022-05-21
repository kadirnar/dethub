def yolov5_predict(img, model_path, device, confidence_threshold):
    from dethub.yolov5_model import Yolov5DetectionModel
    
    model = Yolov5DetectionModel(model_path, confidence_threshold, device)
    model.object_prediction_list(img) 
    model.yolov5_visualization(img)
    return model.prediction_list
    

def torchvision_predict(img, model_path, confidence_threshold):
    from dethub.utils.vis import torchvision_visualization
    from dethub.model_predict import torchvision_prediction
    from dethub.utils.data_utils import read_image

    img = read_image(img)
    img, prediction_boxes, prediction_class, prediction_score = torchvision_prediction(img, model_path, confidence_threshold)
    torchvision_visualization(img, prediction_boxes, prediction_score, prediction_class)
