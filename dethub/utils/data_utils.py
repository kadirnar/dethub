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
    import numpy as np

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

    img_load = tf.keras.preprocessing.image.load_img(image)
    img_array = tf.keras.preprocessing.image.img_to_array(img_load)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.cast(img_array, tf.uint8)
    return img_array


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
