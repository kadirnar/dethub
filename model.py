class DetectionModel():
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        image_size: int = 512,
    ):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.model = None

    def load_model(self):
        raise NotImplementedError()

    def model_predict(self, image, image_size):
        raise NotImplementedError()

    def object_detection_list(self, image):
        raise NotImplementedError()


class Yolov5Model(DetectionModel):
    def load_model(self):

        try:
            import yolov5
        except ImportError:
            raise ImportError(
                "Please install yolov5 with `pip install yolov5`"
            )

        try:

            model = yolov5.load(self.model_path, self.device)
            model.conf = self.confidence_threshold
            self.model = model
        except Exception as e:
            raise TypeError("Model path is not a valid yolov5 model:", e)

    def model_predict(self, image, image_size):
        if self.model is None:
            raise TypeError("Model is not loaded")

        prediction = self.model.predict(image, self.image_size)
        self.prediction = prediction

    def object_predcition_list(self, image):
        if self.prediction is None:
            raise TypeError("Model prediction is not loaded")

        object_predic_list = []

        for _, xyxy in enumerate(self.prediction):
            for pred in xyxy.cpu.detach().numpy():
                x1, y1, x2, y2 = int(pred[0]), int(
                    pred[1]), int(pred[2]), int(pred[3])
                bbox = [x1, y1, x2, y2]
                score = pred[4]
                category_id = int(pred[5])
                category_name = self.model.classes[category_id]
                object_predic_list.append(
                    {
                        "bbox": bbox,
                        "score": score,
                        "category_id": category_id,
                        "category_name": category_name,
                    }
                )
        self.pred_list = object_predic_list
