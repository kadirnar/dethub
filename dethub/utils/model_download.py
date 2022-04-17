import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class Yolov5TestConstants:
    YOLOV5N_MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt"
    YOLOV5N_MODEL_PATH = "dethub/yolov5/yolov5n.pt"


def download_yolov5n_model(destination_path: Optional[str] = None):

    if destination_path is None:
        destination_path = Yolov5TestConstants.YOLOV5N_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov5TestConstants.YOLOV5N_MODEL_URL,
            destination_path,
        )
