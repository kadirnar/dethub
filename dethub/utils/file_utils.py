import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class ModelDownload:
    def __init__(
        self, url: Optional[str] = None, model_path: Optional[str] = None
    ):
        self.url = url
        self.model_path = model_path

    @staticmethod
    def download_from_url(self, from_url: str, to_path: str):

        Path(to_path).parent.mkdir(parents=True, exist_ok=True)

        if not path.exists(to_path):
            urllib.request.urlretrieve(
                from_url,
                to_path,
            )

    @staticmethod
    def yolov5n(self, destination_path: Optional[str] = None):
        YOLOV5N_MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt"
        YOLOV5N_MODEL_PATH = "dethub/models/yolov5/yolov5n.pt"

        if destination_path is None:
            destination_path = YOLOV5N_MODEL_PATH

        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

        if not path.exists(destination_path):
            urllib.request.urlretrieve(
                YOLOV5N_MODEL_URL,
                destination_path,
            )

        return destination_path

    @staticmethod
    def torchvision(self, destination_path: Optional[str] = None):
        TORCHVISION_MODEL_URL = "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
        TORCHVISION_MODEL_PATH = (
            "dethub/models/torchvision/fasterrcnn_resnet50_fpn.pth"
        )

        if destination_path is None:
            destination_path = TORCHVISION_MODEL_PATH

        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

        if not path.exists(destination_path):
            urllib.request.urlretrieve(
                TORCHVISION_MODEL_URL,
                destination_path,
            )

        return destination_path
