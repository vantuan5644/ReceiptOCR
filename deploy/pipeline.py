import matplotlib
import matplotlib.pyplot as plt

from deploy.alignment.alignment import get_alignment_results
from deploy.field_detection.field_detection import get_field_detection_results
from deploy.ocr.ocr import get_ocr_results
matplotlib.use('TkAgg')

if __name__ == "__main__":

    test_image = '/home/vantuan5644/PycharmProjects/ReceiptOCR/datasets/COOP/bill_coop_04/bill_coop_04/img_1.jpg'

    alignment_results = get_alignment_results(test_image)
    """
    alignment_results = {img_path_0: {'image': <np.ndarray>, 'detections': {}, 'transformed_image': <np.ndarray>},
                        img_path_1: {'image': <np.ndarray>, 'detections': {}, 'transformed_image': <np.ndarray>},
                        ...}
    """

    transformed_image = alignment_results[test_image]['transformed_image']

    assert transformed_image is not None, UserWarning(f"Can't align input image {test_image}")

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
    ocr_results = []
    for field in field_detection_results['cropped_fields']:
        for field_name, field_image in field.items():
            ocr_results.append({field_name: get_ocr_results(field_image)})

    """
    ocr_results = [{'bill_code': <str>},
                   {'date': <str>},
                   {'market_name': <str>},
                   {'product_attributes': <str>},
                   ...
                   {'product_attributes': <str>},
                   ]
    """
    print(ocr_results)
