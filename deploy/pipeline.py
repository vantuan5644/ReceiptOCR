import json
import os
from io import BytesIO

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import base64

from PIL import Image

from deploy.alignment.alignment import get_alignment_results
from deploy.field_detection.field_detection import get_field_detection_results
from deploy.ocr.NER import get_NER_results
from deploy.ocr.ocr import get_ocr_results

matplotlib.use('TkAgg')


def padding_image(src):
    """
    @param padding: 'mirror', 'zeros'
    """

    top = int(0.1 * src.shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.1 * src.shape[1])  # shape[1] = cols
    right = left


    value = [0, 0, 0]
    borderType = cv2.BORDER_CONSTANT
    dst = cv2.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)

    return dst

def convert_to_base64(image):
    buff = BytesIO()
    image = Image.fromarray(image)
    image.save(buff, format="JPEG")
    s = base64.b64encode(buff.getvalue()).decode("utf-8")
    return s

def ocr_pipeline(image: np.ndarray):
    alignment_results = get_alignment_results(image)
    """
    alignment_results = {'image': <np.ndarray>, 'detections': {}, 'transformed_image': <np.ndarray>}
    """

    transformed_image = alignment_results['transformed_image']
    # assert transformed_image is not None, UserWarning(f"Can't align input image")
    if transformed_image is None:
        return "fail"

    field_detection_results = get_field_detection_results(transformed_image=transformed_image)
    """
    field_detection_results = {'image': <np.ndarray>, 
                               'detections': {},
                               'cropped_fields': [{'bill_code': <np.ndarray>},
                                                  {'date': <np.ndarray>},
                                                  {'market_name': <np.ndarray>},
                                                  {'product_attributes': <np.ndarray>},
                                                  ...
                                                  {'product_attributes': <np.ndarray>},
                                                  ]
    """

    results = {
        "alignment": convert_to_base64(alignment_results['image']),
        "alignment_crop": convert_to_base64(alignment_results['transformed_image']),
        "field_detection": convert_to_base64(field_detection_results['image']),
        "ocr": {},
    }

    product_attributes = []

    for field in field_detection_results['cropped_fields']:
        for field_name, field_image in field.items():
            if field_name != 'product_attributes':
                results['ocr'][field_name] = {'image': convert_to_base64(field_image), 'text': get_ocr_results(field_image)['single_line']}
            else:
                product_attributes.append({'image': field_image})

    for product in product_attributes:
        text = get_ocr_results(product['image'])['multiple_lines'].replace('\\n', ' ')
        product['image'] = convert_to_base64(product['image'])
        NER_results = get_NER_results(text)
        for key, value in NER_results.items():
            product[key] = value
    results['ocr']['product_attributes'] = product_attributes

    return results

if __name__ == '__main__':
    img = '/home/vantuan5644/PycharmProjects/ReceiptOCR/datasets/COOP/bill_coop_04/bill_coop_04/img_8.jpg'
    img = plt.imread(img)
    print(img.dtype)


    dst_width = 1280
    scale_ratio = dst_width / img.shape[1]
    width = dst_width
    height = int(img.shape[0] * scale_ratio)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    padded = padding_image(resized)

    results = ocr_pipeline(padded)
    with open('data.json', 'w') as f:
        f.write(json.dumps(results))

    # print(convert_to_base64(resized))
