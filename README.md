<div align="center">
<h1>
Object Detection and Classification Model Hub
</h1>
<img src="doc/torchvisin_prediction.jpg" alt="Yolite" width="700">
</div>


## <div align="center">Overview</div>


### 1. Yolov5 Object Prediction List

```
from dethub.yolov5_model import Yolov5DetectionModel
model = Yolov5DetectionModel(model_path, confidence_threshold, device)
model.object_prediction_list(img) 
```
#### Output:

```
{
'bbox': [448, 310, 494, 342], 
'score': 0.73421854, 
'category_name': 'car'
}
```
### 2. Yolov5 Object Visualization


```
from dethub.yolov5_model import Yolov5DetectionModel
model = Yolov5DetectionModel(model_path, confidence_threshold, device)
model.object_prediction_list(img) 
model.yolov5_visualization(img)
```
#### Output:
<img src="doc/yolov5_prediction.jpg" alt="Yolov5" width="800">


### 1. Torchvision Object Prediction List

```
from dethub.yolov5_model import Yolov5DetectionModel
model = Yolov5DetectionModel(model_path, confidence_threshold, device)
model.object_prediction_list(img) 
```
#### Output:
```
[[[(941.08374, 1228.0343), (1012.77856, 1321.7717)], 'truck', 0.8594939], 
[[(2614.4521, 1528.8745), (2845.1682, 1676.0325)], 'car', 0.8239291]]
```
### 2. Torchvision Object Visualization
```
from dethub.yolov5_model import Yolov5DetectionModel
model = Yolov5DetectionModel(model_path, confidence_threshold, device)
model.object_prediction_list(img) 
model.yolov5_visualization(img)
```
#### Output:
<img src="doc/torchvisin_prediction.jpg" alt="Yolov5" width="800">


## TODO
- [ ] Torchvision will simplify.
- [ ] Detectron2 will be added.
- [ ] TfHub will be added.
- [ ] Torch Hub will be added.
- [ ] Hugging Face will be added.
- [ ] Visualization and model files will be made more functional.
