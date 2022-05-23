def yolov5_predict(img, model_path, device, confidence_threshold):
    from dethub.model import Yolov5

    torcvision_model = Yolov5(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    torcvision_model.model
    torcvision_model.object_prediction_list(img)
    torcvision_model.visualization(img)
    return torcvision_model.object_prediction_list(img)
    

def torchvision_predict(img, model_path, confidence_threshold,device):
    from dethub.model import Torchvision

    torcvision_model = Torchvision(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    torcvision_model.model
    torcvision_model.object_prediction_list(img)
    torcvision_model.visualization(img)
    return torcvision_model.object_prediction_list(img)
