from dethub.model import *


def pretrained_weights(model_type):
    """
    Returns the path to the pretrained weights
    args:
        model_type: str

    """
    from dethub.utils.file_utils import ModelDownload

    if model_type == "yolov5":
        ModelDownload.yolov5n()

    elif model_type == "torchvision":
        ModelDownload.torchvision()


def visualizer(image, detection_model):
    """
    Visualize the image with the bounding boxes and labels
    Args:
        image: numpy array
        detection_model: detection model
    """

    import cv2

    from dethub.utils.visualize import vis

    image = cv2.imread(image)
    object_prediction_list = detection_model.object_prediction_list(image)
    vis(image, object_prediction_list)


def run(model_type, model_path, image_path, device="cpu", confidence_threshold=0.5):
    """
    Run the detection model
    Args:
        model_type: str
        model_path: str
        image_path: str
        device: str
        confidence_threshold: float
    """
    if model_type == "yolov5":
        detection_model = Yolov5(model_path, device, confidence_threshold)

    elif model_type == "torchvision":
        detection_model = TorchVision(model_path, device, confidence_threshold)

    elif model_type == "tensorflow":
        detection_model = TensorflowHub(model_path, device, confidence_threshold)

    elif model_type == "YoloXHub":
        detection_model = YoloXHub(model_path, device, confidence_threshold)

    elif model_type == "yolov7hub":
        detection_model = Yolov7Hub(model_path, device, confidence_threshold)

    visualizer(image_path, detection_model)
