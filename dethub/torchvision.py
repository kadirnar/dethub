class TorchvisionModel:
    def __init__(self, model_path, confidence_threshold, device):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.load_model()
        
    def load_model(self):
        import torchvision,torch

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        self.model = model
    
    def object_prediction_list(self, img):
        from dethub.utils.data_utils import read_image, numpy_to_torch, torch_to_numpy, COCO_CLASSES

        img = read_image(img)
        img = numpy_to_torch(img)
        prediction = self.model([img])
        img = torch_to_numpy(img)
        prediction_class = [COCO_CLASSES[i] for i in list(prediction[0]['labels'].numpy())]
        prediction_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(prediction[0]['boxes'].detach().numpy())]
        prediction_score = list(prediction[0]['scores'].detach().numpy())
        prediction_thresh = [prediction_score.index(x) for x in prediction_score if x > self.confidence_threshold][-1]
        prediction_boxes = prediction_boxes[:prediction_thresh + 1]
        prediction_class = prediction_class[:prediction_thresh + 1]
        prediction_list = []
        for i in range(len(prediction_boxes)):
            prediction_list.append([prediction_boxes[i], prediction_class[i], prediction_score[i]])
        self.prediction_list = prediction_list
        return prediction_list
    
    def visualization(self, img):
        import numpy as np
        import cv2
        
        pred_boxes = [i[0] for i in self.prediction_list]

        _COLORS = np.array(
            [
                0.000, 0.447, 0.741,
                0.850, 0.325, 0.098,
                0.929, 0.694, 0.125,
                0.494, 0.184, 0.556,
                0.466, 0.674, 0.188,
                0.301, 0.745, 0.933,
                0.635, 0.078, 0.184,
                0.300, 0.300, 0.300,
                0.600, 0.600, 0.600,
                1.000, 0.000, 0.000,
                1.000, 0.500, 0.000,
                0.749, 0.749, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 1.000,
                0.667, 0.000, 1.000,
                0.333, 0.333, 0.000,
                0.333, 0.667, 0.000,
                0.333, 1.000, 0.000,
                0.667, 0.333, 0.000,
                0.667, 0.667, 0.000,
                0.667, 1.000, 0.000,
                1.000, 0.333, 0.000,
                1.000, 0.667, 0.000,
                1.000, 1.000, 0.000,
                0.000, 0.333, 0.500,
                0.000, 0.667, 0.500,
                0.000, 1.000, 0.500,
                0.333, 0.000, 0.500,
                0.333, 0.333, 0.500,
                0.333, 0.667, 0.500,
                0.333, 1.000, 0.500,
                0.667, 0.000, 0.500,
                0.667, 0.333, 0.500,
                0.667, 0.667, 0.500,
                0.667, 1.000, 0.500,
                1.000, 0.000, 0.500,
                1.000, 0.333, 0.500,
                1.000, 0.667, 0.500,
                1.000, 1.000, 0.500,
                0.000, 0.333, 1.000,
                0.000, 0.667, 1.000,
                0.000, 1.000, 1.000,
                0.333, 0.000, 1.000,
                0.333, 0.333, 1.000,
                0.333, 0.667, 1.000,
                0.333, 1.000, 1.000,
                0.667, 0.000, 1.000,
                0.667, 0.333, 1.000,
                0.667, 0.667, 1.000,
                0.667, 1.000, 1.000,
                1.000, 0.000, 1.000,
                1.000, 0.333, 1.000,
                1.000, 0.667, 1.000,
                0.333, 0.000, 0.000,
                0.500, 0.000, 0.000,
                0.667, 0.000, 0.000,
                0.833, 0.000, 0.000,
                1.000, 0.000, 0.000,
                0.000, 0.167, 0.000,
                0.000, 0.333, 0.000,
                0.000, 0.500, 0.000,
                0.000, 0.667, 0.000,
                0.000, 0.833, 0.000,
                0.000, 1.000, 0.000,
                0.000, 0.000, 0.167,
                0.000, 0.000, 0.333,
                0.000, 0.000, 0.500,
                0.000, 0.000, 0.667,
                0.000, 0.000, 0.833,
                0.000, 0.000, 1.000,
                0.000, 0.000, 0.000,
                0.143, 0.143, 0.143,
                0.286, 0.286, 0.286,
                0.429, 0.429, 0.429,
                0.571, 0.571, 0.571,
                0.714, 0.714, 0.714,
                0.857, 0.857, 0.857,
                0.000, 0.447, 0.741,
                0.314, 0.717, 0.741,
                0.50, 0.5, 0
            ]
        ).astype(np.float32).reshape(-1, 3)

        for i,box in enumerate(pred_boxes):
            score = self.prediction_list[i][2]
            color = (_COLORS[i % len(_COLORS)][::-1] * 255).astype(np.uint8)
            color = (int(color[0]), int(color[1]), int(color[2]))
            text = '{}:{:.1f}%'.format(self.prediction_list[i][1], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[i % len(_COLORS)]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            x, y, w, h = int(box[0][0]), int(box[0][1]), int(box[1][0] - box[0][0]), int(box[1][1] - box[0][1])
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            txt_bk_color = (_COLORS[i % len(_COLORS)][::-1] * 255).astype(np.uint8)
            txt_bk_color = (int(txt_bk_color[0]), int(txt_bk_color[1]), int(txt_bk_color[2]))
            cv2.rectangle(img, (x, y), (x + txt_size[0] + 3, y + txt_size[1] + 3), txt_bk_color, -1)
            cv2.putText(img, text, (x + 1, y + txt_size[1] + 1), font, 0.4, txt_color, 1, cv2.LINE_AA)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
