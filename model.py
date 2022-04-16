from typing import Optional
import numpy as np


class DetectionModel:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.3,
        load_at_init: bool = True,
        image_size: int = None,
    ):

        self.model_path = model_path
        self.model = None
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

        # automatically load model if load_at_init is True
        if load_at_init:
            self.load_model()

    def load_model(self):
        raise NotImplementedError()


class Yolov5DetModel(DetectionModel):
    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        try:
            import yolov5
        except ImportError:
            raise ImportError(
                'Please run "pip install -U yolov5" ' "to install YOLOv5 first for YOLOv5 inference.")

        # set model
        try:
            model = yolov5.load(self.model_path, device=self.device)
            model.conf = self.confidence_threshold
            self.model = model
        except Exception as e:
            raise TypeError("model_path is not a valid yolov5 model path: ", e)
