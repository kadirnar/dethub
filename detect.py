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

def tfhub_predict(img, model_path, device, confidence_threshold, label_file):
    from dethub.model import TfHub

    tfhub_model = TfHub(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold,
        label_file=label_file,
    )
    tfhub_model.model
    tfhub_model.object_prediction_list(img)
    return tfhub_model.object_prediction_list(img)

model_file = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
img_file = "dethub/data/highway1.jpg"
label_file = "dethub/data/coco_label.txt"

pred = tfhub_predict(img_file, model_file, "cpu", 0.5, label_file)
print(pred[0])