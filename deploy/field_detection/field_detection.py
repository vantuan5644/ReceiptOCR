import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

MODEL_PATH = 'weights/yolo/best.pt'
CLASSES = ['market_name', 'date', 'bill_code', 'product_attributes']

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=MODEL_PATH)


def yolov5_predict(image: Image.Image):
    # img = Image.open(image_path)
    results = model(image)
    prediction = []
    for item in results.xyxy[0]:
        [xmin, ymin, xmax, ymax, conf, class_num] = item
        xmin, ymin, xmax, ymax, conf = int(xmin.item()), int(ymin.item()), int(xmax.item()), int(ymax.item()), np.round(
            conf.item(), 4)
        class_name = CLASSES[int(class_num)]
        prediction.append({
            "label": class_name,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax
        })
    return prediction


def imShow(image):
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (3 * width, 3 * height), interpolation=cv2.INTER_CUBIC)
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    # plt.rcParams['figure.figsize'] = [10, 5]
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.show()


def draw_bndbox(image):
    # img = cv2.imread(image_path)
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    color = (255, 0, 0)
    thickness = 1

    for item in yolov5_predict(image):
        xmin, ymin, xmax, ymax, label = item["xmin"], item["ymin"], item["xmax"], item["ymax"], item["label"]
        org = (int(xmin / 2 + xmax / 2), ymin)

        if label == "market_name":
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        elif label == "date":
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        elif label == "bill_code":
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        elif label == "product_attributes":
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255), 2)
        img = cv2.putText(img, label, org, font, fontScale, color, thickness, cv2.LINE_AA)
    return img


def get_field_detection_results(transformed_image):

    detections = yolov5_predict(transformed_image)
    """    
    detections = [{'label': 'bill_code', 'xmin': <int>, 'xmax': <int>, 'ymin': <int>, 'ymax': <int>},
                  {'label': 'date', 'xmin': <int>, 'xmax': <int>, 'ymin': <int>, 'ymax': <int>},
                  {'label': 'market_name', 'xmin': <int>, 'xmax': <int>, 'ymin': <int>, 'ymax': <int>},
                  
                  {'label': 'product_attributes', 'xmin': <int>, 'xmax': <int>, 'ymin': <int>, 'ymax': <int>},
                  {'label': 'product_attributes', 'xmin': <int>, 'xmax': <int>, 'ymin': <int>, 'ymax': <int>},
                  ...
                  ]
    """
    boxes_image = draw_bndbox(transformed_image)

    results = {'image': boxes_image, 'detections': detections, 'cropped_fields': []}

    for field in detections:
        results['cropped_fields'].append({field['label']: transformed_image[field['ymin']:field['ymax'], field['xmin']:field['xmax']]})

    return results
