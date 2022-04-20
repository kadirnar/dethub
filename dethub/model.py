from asyncio.log import logger
from typing import Optional


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
        self.object_predictions = None

        if load_at_init:
            self.load_model()

    def load_model(self):
        """
        Loads the model from the model path.
        """
        NotImplementedError()

    def model_predict(self, image, image_size=None):
        """
        Returns a list of bounding boxes and a list of class names.
        """
        NotImplementedError()


class Yolov5DetectionModel(DetectionModel):
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
            import logger
            model = yolov5.load(self.model_path, device=self.device)
            model.conf = self.confidence_threshold
            self.model = model
            logger.info(f"YOLOv5 model loaded from {self.model_path}.")
        except Exception as e:
            TypeError("model_path is not a valid yolov5 model path: ", e)

    def model_predict(self, image, image_size=None):
        """
        Returns a list of bounding boxes and a list of class names.
        """

        try:
            import yolov5
        except ImportError:
            raise ImportError(
                'Please run "pip install -U yolov5" ' "to install YOLOv5 first for YOLOv5 inference.")
        logger.info(f"Predicting with YOLOv5 model.")

        if self.model is None:
            raise ValueError("model is not loaded.")

        if image_size is not None:
            model_prediction = self.model(image, self.image_size)
        else:
            model_prediction = self.model(image)

        self.object_predictions = model_prediction
