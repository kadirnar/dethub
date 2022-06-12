from regex import P
from transformers import AutoModelForObjectDetection, AutoConfig
from utils.data_utils import numpy_to_torch, imshow
import torch
import cv2
from postprocces.utils import post_process
from pybboxes import BoundingBox
from PIL import Image


threshold = 0.5
img = cv2.imread("dethub/data/highway1.jpg")
img = numpy_to_torch(img).unsqueeze(0)
target_sizes = torch.tensor([tuple(reversed(img.shape[-2:]))])


config = AutoConfig.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_config(config)

outputs = model(img)
output_dict = post_process(outputs, target_sizes)[0]
img = img.squeeze().permute(1, 2, 0).cpu().numpy()

scores = output_dict["scores"].detach().cpu().numpy()
boxes = output_dict["boxes"].detach().cpu().numpy()
labels = output_dict["labels"].detach().cpu().numpy()


