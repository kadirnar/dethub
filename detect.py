def yolov5_predict(img, model_path, device, confidence_threshold):
    from dethub.yolov5 import Yolov5DetectionModel

    model = Yolov5DetectionModel(model_path, confidence_threshold, device)
    model.object_prediction_list(img) 
    model.yolov5_visualization(img)
    return model.prediction_list
    

def torchvision_predict(img, model_path, confidence_threshold,device):
    from dethub.torchvision import TorchvisionModel

    model = TorchvisionModel(model_path, confidence_threshold, device)
    model.object_prediction_list(img)
    model.torchvision_visualization(img)
    return model.prediction_list
