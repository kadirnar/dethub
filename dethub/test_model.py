import unittest
from utils.model_download import Yolov5TestConstants, download_yolov5n_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 320


class TestYolov5Model(unittest.TestCase):
    def test_load_model(self):
        from model import Yolov5DetModel

        download_yolov5n_model()

        yolov5_detection_model = Yolov5DetModel(
            model_path=Yolov5TestConstants.YOLOV5N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=True,
        )

        self.assertNotEqual(yolov5_detection_model.model, None)


if __name__ == "__main__":
    unittest.main()
