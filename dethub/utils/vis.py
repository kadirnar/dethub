import cv2

def yolov5_visualization(img, prediction_list):
    for pred in prediction_list:
        x1, y1, x2, y2 = pred['bbox']
        score = pred['score']
        category_name = pred['category_name']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(img, (x1, y1-10), (x1 + 70, y1), (0, 0, 255), -1)
        cv2.putText(img, f"{category_name} {score:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255), 1)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return img
    
def torchvision_visualization(img, prediction_boxes, prediction_score, prediction_class):
    for i, box in enumerate(prediction_boxes):
        score = prediction_score[i]
        labels = '%{} {}'.format(float(int(score * 100)), prediction_class[i])
        x, y, w, h = int(box[0][0]), int(box[0][1]), int(box[1][0] - box[0][0]), int(box[1][1] - box[0][1])
        cv2.rectangle(img, (x, y-10), (x + 80, y), (0, 0, 255), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(img, labels, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img
