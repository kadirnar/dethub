<div align="center">
<h1>
DetHub: Object Detection Model Hub
</h1>
<img src="doc/torchvisin_prediction.jpg" alt="Yolite" width="700">
</div>

### Yolov5 Object Prediction and Visualization
```
import cv2        
img = cv2.imread('dethub/data/highway1.jpg')   
model_path = "models/torchvision/fasterrcnn_resnet50_fpn.pth"
torcvision_model = Torchvision(model_path=model_path,device='cpu'confidence_threshold=0.5,)
torcvision_model.model
torcvision_model.object_prediction_list(img)
torcvision_model.visualization(img) 
```
#### Output:
```
{
'bbox': [448, 310, 494, 342], 
'score': 0.73421854, 
'category_name': 'car'
}
```
<img src="doc/yolov5_prediction.jpg" alt="Yolov5" width="800">

### Torchvision Object Prediction and Visualization
```
import cv2
img = cv2.imread('dethub/data/highway1.jpg')   
model_file = "models/torchvision/fasterrcnn_resnet50_fpn.pth"
pred = torchvision_predict(img, model_file, 0.5, "cpu")
```
#### Output:
```
[[[(941.08374, 1228.0343), (1012.77856, 1321.7717)], 'truck', 0.8594939], 
[[(2614.4521, 1528.8745), (2845.1682, 1676.0325)], 'car', 0.8239291]]
```
<img src="doc/torchvisin_prediction.jpg" alt="Yolov5" width="800">

## TODO
- [ ] Torchvision will simplify.
- [ ] Detectron2 will be added.
- [ ] TfHub will be added.
- [ ] Torch Hub will be added.
- [ ] Hugging Face will be added.
- [ ] Visualization and model files will be made more functional.

References:
- [SAHI](https://github.com/obss/sahi)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)