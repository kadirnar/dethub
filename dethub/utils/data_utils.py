import numpy as np


def numpy_to_torch(img):
    """
    Convert numpy array to torch tensor
    Args:
        img: numpy array
    """
    import torch

    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    if img.max() > 1:
        img /= 255
    return img


def torch_to_numpy(img):
    """
    Convert torch tensor to numpy array
    Args:
        img: numpy array
    """
    img = img.numpy()
    if img.max() > 1:
        img /= 255
    img = img.transpose((1, 2, 0))
    return img


def read_image(img):
    """
    Read image from path
    Args:
        img: numpy array
    """
    import cv2

    color_conversion = {
        "grayscale:": 2,
        "rgb": 3,
        "rgba": 4,
    }
    if type(img) is str:
        img = cv2.imread(img)

    elif type(img) is bytes:
        nparr = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(img) is np.ndarray:
        if len(img.shape) is color_conversion["grayscale:"]:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        elif (
            len(img.shape) is color_conversion["rgb"] and img.shape[2] is color_conversion["rgba"]
        ):  # RGBA
            img = img[:, :, :3]

    return img


def create_dir(_dir):
    """
    Create directory if it doesn't exist
    Args:
        _dir: str
    """
    import os

    if not os.path.exists(_dir):
        os.makedirs(_dir)


def download(url: str, save_path: str):
    """
    Download file from url
    Args:
        url: str
        save_path: str
    """
    import os

    import gdown

    create_dir(os.path.dirname(save_path))
    gdown.download(url, save_path, quiet=False)


def imshow(img):
    """
    Display image
    Args:
        img: numpy array
    """
    import cv2

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def tf_resize(array, size):
    """
    Resize image using tensorflow
    Args:
        array: numpy array
        size: int
    """
    import tensorflow as tf

    return tf.image.resize(array, [size, size]).numpy()


def to_float_tensor(img: np.ndarray):
    """
    Convert image to float tensor
    Args:
        image: numpy array
    """
    import tensorflow as tf

    float_image = np.asarray(img, np.float32)
    if img.max() <= 1:
        float_image = float_image * 255.0
    image_tensor = tf.convert_to_tensor([np.asarray(float_image, np.uint8)], tf.uint8)
    return image_tensor


def tensorflow_resize(array, size):
    """
    Resize image using tensorflow
    Args:
        array: numpy array
        size: int
    """
    import tensorflow as tf

    return tf.image.resize(array, [size, size]).numpy()


COCO_CLASSES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

TF_COCO_CLLASES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
