from transformers import AutoConfig, AutoModelForObjectDetection
from torchvision import transforms as T
from PIL import Image
import requests


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(url, stream=True).raw)
img = T.ToTensor()(img).unsqueeze(0)

config = AutoConfig.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_config(config)

outputs = model(img)

bbox = outputs.pred_boxes
logits = outputs.logits
last_hidden_states = outputs.last_hidden_state
encoder_last_hidden_state = outputs.encoder_last_hidden_state
id2label = config.id2label















