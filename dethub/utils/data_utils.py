import numpy as np


def numpy_to_torch(img):
    import torch

    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    if img.max() > 1:
        img /= 255
    return img


def torch_to_numpy(img):
    img = img.numpy()
    if img.max() > 1:
        img /= 255
    img = img.transpose((1, 2, 0))
    return img


def read_image(img):
    import cv2

    if type(img) == str:
        img = cv2.imread(img)

    elif type(img) == bytes:
        nparr = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(img) == np.ndarray:
        if len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = img

        elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]

    return img


def create_dir(_dir):
    import os

    if not os.path.exists(_dir):
        os.makedirs(_dir)


def download(url: str, save_path: str):
    import os

    import gdown

    create_dir(os.path.dirname(save_path))
    gdown.download(url, save_path, quiet=False)


def imshow(img):
    import cv2

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(array, size):
    import tensorflow as tf

    return tf.image.resize(array, [size, size]).numpy()


def to_float_tensor(image: np.ndarray):
    import tensorflow as tf

    float_image = np.asarray(image, np.float32)
    if image.max() <= 1:
        float_image = float_image * 255.0
    image_tensor = tf.convert_to_tensor(
        [np.asarray(float_image, np.uint8)], tf.uint8
    )
    return image_tensor


def tf_resize(array, size):
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
