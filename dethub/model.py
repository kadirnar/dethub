from typing import Optional

from dethub.utils.data_utils import imshow
from dethub.utils.visualize import vis


class DetectionModel:
    def __init__(
        self,
        model_path: str,
        device: str,
        confidence_threshold: float = 0.5,
        label_file: Optional[str] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.label_file = label_file
        self.model = None
        self.load_model()

    def get_image(self, img):
        """
        Returns a tensor of the image.
        """
        raise NotImplementedError

    def get_label(self):
        """
        Returns a list of labels for the given label file.
        """
        raise NotImplementedError

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

    def visualization(self, img):
        """
        Returns a visualization of the predictions for the given image.
        """
        raise NotImplementedError


class Yolov5(DetectionModel):
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
                category_id = pred[5]
                prediction_list.append(
                    {"bbox": bbox, "score": score, "category_name": category_name, "category_id": category_id}
                )

        self.prediction_list = prediction_list

        
class Torchvision(DetectionModel):
    def load_model(self):
        import torch
        import torchvision

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        self.model = model

    def object_prediction_list(self, img):
        from dethub.utils.data_utils import COCO_CLASSES, numpy_to_torch, read_image, torch_to_numpy

        img = read_image(img)
        img = numpy_to_torch(img)
        prediction = self.model([img])
        img = torch_to_numpy(img)
        prediction_class = [COCO_CLASSES[i] for i in list(prediction[0]["labels"].numpy())]
        prediction_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(prediction[0]["boxes"].detach().numpy())]
        prediction_score = list(prediction[0]["scores"].detach().numpy())
        prediction_thresh = [prediction_score.index(x) for x in prediction_score if x > self.confidence_threshold][-1]
        prediction_boxes = prediction_boxes[: prediction_thresh + 1]
        prediction_class = prediction_class[: prediction_thresh + 1]
        prediction_list = []

        for i in range(len(prediction_boxes)):
            prediction_list.append([prediction_boxes[i], prediction_class[i], prediction_score[i]])
        self.prediction_list = prediction_list

        
class TensorflowHub(DetectionModel):
    def load_model(self):
        import tensorflow as tf
        import tensorflow_hub as hub

        with tf.device(self.device):
            self.model = hub.load(self.model_path)

    def object_prediction_list(self, image):
        import cv2
        import tensorflow as tf

        from dethub.utils.data_utils import COCO_CLASSES, resize, to_float_tensor

        img = to_float_tensor(image)

        category_mapping = {str(i): COCO_CLASSES[i] for i in range(len(COCO_CLASSES))}
        img = to_float_tensor(image)
        npy_img = cv2.imread(image)
        prediction_result = self.model(img)

        self.image_height, self.image_width = img.shape[0], img.shape[1]
        boxes = prediction_result["detection_boxes"][0].numpy()
        scores = prediction_result["detection_scores"][0].numpy()
        category_ids = prediction_result["detection_classes"][0].numpy()
        with tf.device(self.device):
            for i in range(min(boxes.shape[0], 100)):
                if scores[i] >= self.confidence_threshold:
                    score = float(scores[i])
                    category_id = int(category_ids[i])

                    category_names = category_mapping[str(category_id - 1)]
                    box = [float(box) for box in boxes[i]]
                    x1, y1, x2, y2 = (
                        int(box[1] * self.image_width),
                        int(box[0] * self.image_height),
                        int(box[3] * self.image_width),
                        int(box[2] * self.image_height),
                    )
                    bbox = [x1, y1, x2, y2]
