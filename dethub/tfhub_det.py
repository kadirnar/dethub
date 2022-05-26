import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def set_device():
    if tf.test.is_gpu_available():
        tf.device("/gpu:0")
    else:
        tf.device("/cpu:0")
        
def get_image():
    img_file = "dethub/data/highway1.jpg"
    img_load = tf.keras.preprocessing.image.load_img(img_file, target_size=(512, 512))
    img_array = tf.keras.preprocessing.image.img_to_array(img_load)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.cast(img_array, tf.uint8)
    return img_array

def detector():
    img_array = get_image()
    detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")
    detector_output = detector(img_array)
    boxes = detector_output["detection_boxes"][0]
    scores = detector_output["detection_scores"][0]
    class_names = detector_output["detection_classes"][0]
    return boxes, scores, class_names

def get_label():
    label_file = "dethub/data/coco_label.txt"
    label = []
    for line in open(label_file):
        label.append(line.strip())    
    return label 
   
def vis():
    boxes, scores, class_names = detector()
    img_array = get_image()

    for i in range(boxes.shape[0]):
        if scores[i] > 0.5:
            box = boxes[i]
            label = get_label()[int(class_names[i] - 1)]
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * 512, xmax * 512, ymin * 512, ymax * 512)
            plt.imshow(img_array[0])
            plt.gca().add_patch(plt.Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor='red', linewidth=3))
            plt.gca().text(left, top, label, bbox=dict(facecolor='blue', alpha=0.5))
    plt.show()
