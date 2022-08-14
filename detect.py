from dethub.model import *
import argparse

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



def parse_arguments():
    """
    Parse the command line arguments
    """
    parser = argparse.ArgumentParser(description="Run the detection model")
    parser.add_argument(
        "--model_type",
        type=str,
        default="yolov5",
        help="Model type",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="dethub/models/yolov5s.pt",
        help="Path to the model",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="data/images/highway.jpg",
        help="Path to the image",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--category_mapping",
        type=str,
        default="dethub/utils/model.yaml",
        help="Path to the category mapping",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=640,
        help="Image size",
    )
    return parser.parse_args()

def run(args):
    """
    Run the detection model
    Args:
        model_type: str
        model_path: str
        image_path: str
        device: str
        confidence_threshold: float
    """
    
    if args.model_type == "yolov5":
        detection_model = Yolov5(args.model_path, args.device, args.confidence_threshold, args.category_mapping, args.image_size)
    
    elif args.model_type == "torchvision":
        detection_model = TorchVision(args.model_path, args.device, args.confidence_threshold, args.category_mapping, args.image_size)

    elif args.model_type == "tensorflow":
        detection_model = TensorflowHub(args.model_path, args.device, args.confidence_threshold, args.category_mapping, args.image_size)

    elif args.model_type == "YoloXHub":
        detection_model = YoloXHub(args.model_path, args.device, args.confidence_threshold, args.category_mapping, args.image_size)

    elif args.model_type == "yolov7hub":
        detection_model = Yolov7Hub(args.model_path, args.device, args.confidence_threshold, args.category_mapping, args.image_size)

    visualizer(args.image_path, detection_model)
    


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
