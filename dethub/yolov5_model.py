class Yolov5DetectionModel:
    def __init__(self, model_path, confidence_threshold, device):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.load_model()

    def load_model(self):
        import yolov5
        model = yolov5.load(self.model_path, device=self.device)
        model.conf = self.confidence_threshold
        self.model = model


    def object_prediction_list(self, img):
        prediction = self.model(img)
        prediction_list = []
        for _, image_predictions_in_xyxy_format in enumerate(prediction.xyxy):
            for pred in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1, y1, x2, y2 = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])
                bbox = [x1, y1, x2, y2]
                score = pred[4]
                category_name = self.model.names[int(pred[5])]
                prediction_list.append({'bbox': bbox, 'score': score, "category_name": category_name})

        self.prediction_list = prediction_list
        
    def yolov5_visualization(self, img):
        import cv2
        
        prediction_boxes = [pred['bbox'] for pred in self.prediction_list]
        for pred in prediction_boxes:
            x1, y1, x2, y2 = pred
            score = self.prediction_list[0]['score']
            label = self.prediction_list[0]['category_name']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y1-10), (x1 + 70, y1), (0, 0, 255), -1)
            cv2.putText(img, f"{label} {score:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255), 1)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        