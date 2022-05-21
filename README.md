<div align="center">
<h1>
Object Detection and Classification Model Hub
</h1>
<img src="doc/pytorch.jpg" alt="Yolite" width="700">
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

## TODO
- [ ] Torchvision will simplify.
- [ ] Detectron2 will be added.
- [ ] TfHub will be added.
- [ ] Torch Hub will be added.
- [ ] Hugging Face will be added.
