class Yolov5DetectionModel:
    def __init__(self, model_path, confidence_threshold, device):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.prediction_list = None
        self.model = None
        self.load_model()

    def load_model(self):
        import yolov5
        model = yolov5.load(self.model_path, device=self.device)
        model.conf = self.confidence_threshold
        self.model = model


    def object_prediction_list(self, image):
        prediction = self.model(image)
        prediction_list = []
        for _, image_predictions_in_xyxy_format in enumerate(prediction.xyxy):
            for pred in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
                bbox = [x1, y1, x2, y2]
                score = pred[4]
                category_name = self.model.names[int(pred[5])]
                prediction_list.append({'bbox': bbox, 'score': score, "category_name": category_name})

        return prediction_list
