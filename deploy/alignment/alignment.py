import os
from typing import List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util, label_map_util
from object_detection.utils import visualization_utils as viz_utils
from tqdm import tqdm

from .utils import get_model_detections, nms, load_image_into_numpy_array, is_completed_prediction, \
    get_transformed_image

matplotlib.use('TkAgg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)


class AlignmentModel():
    def __init__(self, model_dir, category_index):
        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        checkpoint_dir = os.path.join(model_dir, 'checkpoint')

        config_path = os.path.join(model_dir, 'pipeline.config')

        configs = config_util.get_configs_from_pipeline_file(config_path)  # importing config
        model_config = configs['model']  # recreating model config
        self.detection_model = model_builder.build(model_config=model_config, is_training=False)  # importing model

        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(checkpoint_dir, 'ckpt-0')).expect_partial()

        self.category_index = category_index

    def get_alignment_result(self, image, return_image=False, box_th=0.25, nms_th=0.5):

        detection_results = {}

        # image_np = load_image_into_numpy_array(image_path)
        detections = get_model_detections(self.detection_model, image)

        key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']
        if box_th:  # filtering detection if a confidence threshold for boxes was given as a parameter
            for key in key_of_interest:
                scores = detections['detection_scores']
                current_array = detections[key]
                filtered_current_array = current_array[scores > box_th]
                detections[key] = filtered_current_array

        if nms_th:  # filtering rectangles if nms threshold was passed in as a parameter
            # creating a zip object that will contain model output info as

            output_info = list(zip(detections['detection_boxes'],
                                   detections['detection_scores'],
                                   detections['detection_classes']
                                   )
                               )
            boxes, scores, classes = nms(output_info)

            detections['detection_boxes'] = np.array(boxes)  # format: [y1, x1, y2, x2]
            detections['detection_scores'] = np.array(scores)
            detections['detection_classes'] = np.array(classes)

        image_h, image_w, _ = image.shape

        result_ = []
        for b, s, c in zip(detections['detection_boxes'], detections['detection_scores'],
                           detections['detection_classes']):
            y1abs, x1abs = b[0] * image_h, b[1] * image_w
            y2abs, x2abs = b[2] * image_h, b[3] * image_w
            result_.append({'bounding_box': np.array([x1abs, y1abs, x2abs, y2abs]), 'class': c + 1, 'score': s})
        detection_results['detections'] = result_

        if return_image:
            label_id_offset = 1
            image_np_with_detections = image.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=box_th,
                agnostic_mode=False,
                line_thickness=5)
            detection_results['image'] = image_np_with_detections

        return detection_results


model_dir = "pretrained_models/alignment/efficientdet_d1_coco17_tpu-32"
label_map = "pretrained_models/alignment/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(label_map, use_display_name=True)
align = AlignmentModel(model_dir=model_dir, category_index=category_index)


def get_alignment_results(image: np.ndarray):
    results = align.get_alignment_result(image, return_image=True)


    # print(result['detections'])
    results['transformed_image'] = None

    if is_completed_prediction(results['detections'], category_index):
        img_transformed = get_transformed_image(results['image'], results['detections'], category_index)
        # dst_path = os.path.join('datasets', 'COOP', 'transformed', os.path.split(image_path)[1])
        # plt.imsave(dst_path, img_transformed)
        results['transformed_image'] = img_transformed

    """
    results = {'image': <np.ndarray>, 'detections': {}, 'transformed_image': <np.ndarray>}
    """

    return results


