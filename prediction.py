from dethub.utils.file_utils import ModelDownload


def yolov5(img="data/highway1.jpg", model_path=ModelDownload.yolov5n(), device="cpu", confidence_threshold=0.5):
    import cv2

    from dethub.model import Yolov5

    img = cv2.imread(img)
    yolov5_model = Yolov5(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    yolov5_model.model
    yolov5_model.object_prediction_list(img)
    yolov5_model.visualization(img)


def torchvision(img, model_path, device, confidence_threshold):
    from dethub.model import Torchvision

    torcvision_model = Torchvision(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold,
    )

    torcvision_model.model
    torcvision_model.object_prediction_list(img)
    torcvision_model.visualization(img)


def tensorflow(img, model_path, device="cpu", confidence_threshold=0.5):
    from dethub.model import TensorflowHub

    label_file = "data/coco_label.txt"

    tfhub_model = TensorflowHub(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold,
        label_file=label_file,
    )
    tfhub_model.model
    tfhub_model.object_prediction_list(img)
    tfhub_model.visualization(img)
