class DetectionModel:
    def __init__(
        self,
        model_path: str,
        device: str,
        confidence_threshold: float = 0.5,
    ):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.load_model()
        self.prediction_list = None

    def load_model(self):
        """
        Loads the model from the path specified in the constructor.
        """
        raise NotImplementedError

    def object_prediction_list(self, img):
        """
        Returns a list of predictions for the given image.
        """
        raise NotImplementedError


class Yolov5(DetectionModel):
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
                x1, y1, x2, y2 = (
                    int(pred[0]),
                    int(pred[1]),
                    int(pred[2]),
                    int(pred[3]),
                )
                bbox = [x1, y1, x2, y2]
                score = pred[4]
                category_name = self.model.names[int(pred[5])]
                category_id = pred[5]
                prediction_list.append(
                    {
                        "bbox": bbox,
                        "score": score,
                        "category_name": category_name,
                        "category_id": category_id,
                    }
                )

        self.prediction_list = prediction_list
        return prediction_list


class TorchVision(DetectionModel):
    def load_model(self):
        import torch
        import torchvision

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        self.model = model

    def object_prediction_list(self, image):
        import numpy as np

        from dethub.utils.data_utils import COCO_CLASSES, numpy_to_torch

        category_names = {
            str(i): COCO_CLASSES[i] for i in range(len(COCO_CLASSES))
        }
        image = numpy_to_torch(image)
        image = image.to(self.device)
        prediction = self.model([image])

        prediction_list = []
        for image_predictions in prediction:

            # get indices of boxes with score > confidence_threshold
            scores = image_predictions["scores"].cpu().detach().numpy()
            selected_indices = np.where(scores > self.confidence_threshold)[0]

            # parse boxes, masks, scores, category_ids from predictions
            category_ids = list(
                image_predictions["labels"][selected_indices]
                .cpu()
                .detach()
                .numpy()
            )
            boxes = list(
                image_predictions["boxes"][selected_indices]
                .cpu()
                .detach()
                .numpy()
            )
            scores = scores[selected_indices]
            for ind in range(len(boxes)):
                bbox = boxes[ind]
                category_id = int(category_ids[ind])
                category_name = category_names[str(int(category_ids[ind]))]
                score = scores[ind]
                prediction_list.append(
                    {
                        "bbox": bbox,
                        "score": score,
                        "category_name": category_name,
                        "category_id": category_id,
                    }
                )

        self.prediction_list = prediction_list
        return prediction_list


class TensorflowHub(DetectionModel):
    def load_model(self):
        import tensorflow as tf
        import tensorflow_hub as hub

        with tf.device(self.device):
            self.model = hub.load(self.model_path)

    def object_prediction_list(self, image):
        import tensorflow as tf

        from dethub.utils.data_utils import TF_COCO_CLLASES, to_float_tensor

        img = to_float_tensor(image)
        prediction_result = self.model(img)

        image_height, image_width = image.shape[0], image.shape[1]
        img = to_float_tensor(image)

        category_mapping = {
            str(i): TF_COCO_CLLASES[i] for i in range(len(TF_COCO_CLLASES))
        }
        img = to_float_tensor(image)
        prediction_result = self.model(img)

        boxes = prediction_result["detection_boxes"][0].numpy()
        scores = prediction_result["detection_scores"][0].numpy()
        category_ids = prediction_result["detection_classes"][0].numpy()
        prediction_list = []
        with tf.device(self.device):
            for i in range(min(boxes.shape[0], 100)):
                if scores[i] >= self.confidence_threshold:
                    score = float(scores[i])
                    category_id = int(category_ids[i])
                    category_names = category_mapping[str(category_id - 1)]
                    box = [float(box) for box in boxes[i]]
                    x1, y1, x2, y2 = (
                        int(box[1] * image_width),
                        int(box[0] * image_height),
                        int(box[3] * image_width),
                        int(box[2] * image_height),
                    )
                    bbox = [x1, y1, x2, y2]
                    prediction_list.append(
                        {
                            "bbox": bbox,
                            "score": score,
                            "category_name": category_names,
                            "category_id": category_id,
                        }
                    )
        self.prediction_list = prediction_list
        return prediction_list


class Yolov5Hub(DetectionModel):
    def load_model(self):
        import torch

        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=self.model_path,
            device=self.device,
        )
        model.conf = self.confidence_threshold
        self.model = model

    def object_prediction_list(self, image):
        prediction = self.model(image)
        prediction_list = []

        for _, image_predictions_in_xyxy_format in enumerate(prediction.xyxy):
            for pred in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1, y1, x2, y2 = (
                    int(pred[0]),
                    int(pred[1]),
                    int(pred[2]),
                    int(pred[3]),
                )
                bbox = [x1, y1, x2, y2]
                score = pred[4]
                category_name = self.model.names[int(pred[5])]
                category_id = pred[5]
                prediction_list.append(
                    {
                        "bbox": bbox,
                        "score": score,
                        "category_name": category_name,
                        "category_id": category_id,
                    }
                )

        self.prediction_list = prediction_list
        return prediction_list
