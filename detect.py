from dethub.model import *


def pretrained_weights(model_type):
    from dethub.utils.file_utils import ModelDownload

    if model_type == "yolov5":
        ModelDownload.yolov5n()

    elif model_type == "torchvision":
        ModelDownload.torchvision()


def get_prediction(image, detection_model):
    import cv2

    from dethub.utils.visualize import vis

    image = cv2.imread(image)
    object_prediction_list = detection_model.object_prediction_list(image)
    vis(image, object_prediction_list)


def run(
    model_type, model_path, image_path, device="cpu", confidence_threshold=0.5
):
    if model_type == "yolov5":
        detection_model = Yolov5(model_path, device, confidence_threshold)

    elif model_type == "torchvision":
        detection_model = TorchVision(model_path, device, confidence_threshold)

    elif model_type == "tensorflow":
        detection_model = TensorflowHub(
            model_path, device, confidence_threshold
        )

    elif model_type == "yolov5hub":
        detection_model = Yolov5Hub(model_path, device, confidence_threshold)
    
    elif model_type == "yolov7hub":
        detection_model = Yolov7Hub(model_path, device, confidence_threshold)

    get_prediction(image_path, detection_model)


# run('yolov5', 'dethub/models/yolov5/yolov5n.pt', 'data/highway1.jpg')
# run("torchvision", "dethub/models/torchvision/fasterrcnn_resnet50_fpn.pth", "data/highway1.jpg")
# run('tensorflow', 'https://tfhub.dev/tensorflow/efficientdet/d3/1', 'data/highway1.jpg')
# run('yolov5hub', 'dethub/models/yolov5/yolov5n.pt', 'data/highway1.jpg', 'cuda:0', 0.5)
run('yolov7hub', 'yolov7.pt', 'data/highway1.jpg', 'cpu', 0.5)