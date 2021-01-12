import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def get_model_detections(detection_model, image_np):
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(detection_model, input_tensor)

    # checking how many detections we got
    num_detections = int(detections.pop('num_detections'))

    # filtering out detection in order to get only the one that are indeed detections
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # defining what we need from the resulting detection dict that we got from model output
    key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']

    # detections['num_detections'] = num_detections

    # filtering out detection dict in order to get only boxes, classes and scores
    detections = {key: value for key, value in detections.items() if key in key_of_interest}

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    return detections


def detect_fn(detection_model, image):
    """
    Detect objects in image.

    Args:
      detection_model: tf model
      image: (tf.tensor): 4D input image

    Returs:
      detections (dict): predictions that model made
    """

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      numpy array with shape (img_height, img_width, 3)
    """

    return np.array(Image.open(path))


def nms(rects, thd=0.5):
    """
    Filter rectangles
    rects is array of oblects ([x1,y1,x2,y2], confidence, class)
    thd - intersection threshold (intersection divides min square of rectange)
    """
    out = []

    remove = [False] * len(rects)

    for i in range(0, len(rects) - 1):
        if remove[i]:
            continue
        inter = [0.0] * len(rects)
        for j in range(i, len(rects)):
            if remove[j]:
                continue
            inter[j] = intersection(rects[i][0], rects[j][0]) / min(square(rects[i][0]), square(rects[j][0]))

        max_prob = 0.0
        max_idx = 0
        for k in range(i, len(rects)):
            if inter[k] >= thd:
                if rects[k][1] > max_prob:
                    max_prob = rects[k][1]
                    max_idx = k

        for k in range(i, len(rects)):
            if (inter[k] >= thd) & (k != max_idx):
                remove[k] = True

    for k in range(0, len(rects)):
        if not remove[k]:
            out.append(rects[k])

    boxes = [box[0] for box in out]
    scores = [score[1] for score in out]
    classes = [cls[2] for cls in out]
    return boxes, scores, classes


def intersection(rect1, rect2):
    """
    Calculates square of intersection of two rectangles
    rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
    return: square of intersection
    """
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
    overlapArea = x_overlap * y_overlap;
    return overlapArea


def square(rect):
    """
    Calculates square of rectangle
    """
    return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts, ordered=True):
    # obtain a consistent order of the points and unpack them
    # individually
    if not ordered:
        rect = order_points(pts)
    else:
        rect = np.array(pts).reshape(4, 2).astype(np.float32)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def max_x_y(points):
    max_ = points['bottom_right']
    for _, point in points.items():
        if point[0] + point[1] > max_[0] + max_[1]:
            max_ = point
    return max_


def flip_to_vertical(points):
    bottom_right_point = max_x_y(points)
    if np.array_equal(bottom_right_point, points['bottom_right']):
        return None
    elif np.array_equal(bottom_right_point, points['top_right']):
        return cv2.cv2.ROTATE_90_COUNTERCLOCKWISE
    elif np.array_equal(bottom_right_point, points['bottom_left']):
        return cv2.cv2.ROTATE_90_CLOCKWISE
    elif np.array_equal(bottom_right_point, points['top_left']):
        return cv2.cv2.ROTATE_180
    else:
        return None


def align_image_4points(image_src, pts1, offset_lr=50, offset_tb=10):
    h_w = cv2.minAreaRect(pts1)[1]
    height, width = int(max(h_w)), int(min(h_w))

    img_w, img_h = image_src.shape[0], image_src.shape[1]
    pts2 = np.float32([[0 + offset_lr, 0 + offset_tb], [img_w - 1 - offset_lr, 0 + offset_tb],
                       [img_w - 1 - offset_lr, img_h - 1 - offset_tb], [0 + offset_lr, img_h - 1 - offset_tb]])

    #    print(np.float32(pts1), pts2.dtype)
    M = cv2.getPerspectiveTransform(np.float32(pts1), pts2)
    transformed_img = cv2.warpPerspective(image_src, M, (img_w, img_h))

    transformed_img = cv2.resize(transformed_img, (width, height))

    return transformed_img


def get_transformed_image(img, prediction_results, category_index):
    points = {}
    for i, box in enumerate(prediction_results):
        name = category_index[box['class']]['name']
        if name != 'receipt':
            coords = np.array(np.hsplit(box['bounding_box'], 2)).mean(axis=0)
            points[name] = coords.astype(int)
        else:
            points[name] = box['bounding_box'].astype(int)

    # flip_arg = flip_to_vertical(points)
    # if flip_arg is not None:
    #     img = cv2.rotate(img, flip_arg)
    #     # Map points to the new coordinate
    #
    # px_offset = 30
    # points['top_left'] -= px_offset
    # points['top_right'][0] += px_offset
    # points['top_right'][1] -= px_offset
    # points['bottom_left'][0] -= px_offset
    # points['bottom_left'][1] += px_offset
    # points['bottom_right'] += px_offset
    # four_points = np.array([points['top_left'], points['top_right'],
    #                         points['bottom_right'], points['bottom_left']])
    # img_transformed = four_point_transform(img, four_points, ordered=False)
    ordered_points = np.array([points['top_left'], points['top_right'], points['bottom_right'], points['bottom_left']])
    img_transformed = align_image_4points(img, ordered_points, offset_lr=30, offset_tb=10)

    return img_transformed


def is_completed_prediction(prediction_result, category_index):
    detected_classes = set([box['class'] for box in prediction_result])
    return True if detected_classes == set(category_index.keys()) else False
