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
    

