import os

import cv2
import numpy as np

from config import PROJECT_ROOT

BINARY_THREHOLD = 200

os.chdir(PROJECT_ROOT)


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (3, 3), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255,  cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(img):
    filtered = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


def convertImageToBinary(path):
    for image in os.listdir(path):
        if 'txt' not in image:
            print(os.path.join(path, image))
            img = cv2.imread(os.path.join(path, image), 0)
            img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
            img = remove_noise_and_smooth(img)
            kernel = np.ones((2, 2), np.uint8)
            img = cv2.bitwise_not(img)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            img = cv2.bitwise_not(img)
            dst = os.path.join(path.replace('original', 'processed'), image)
            cv2.imwrite(dst, img)


if __name__ == '__main__':
    convertImageToBinary('datasets/OCR/original')

# def anno(path):
#     f = open("annotations.txt","w+")
#     for text in os.listdir(path):
#         if 'txt' in text:
#             with open(os.path.join(path,text),"r") as ff:
#                 line = ff.read()
#             f.write(f"grount_truth/{text[:-7]}.jpg {line}\n")
#     f.close()
# path = 'transformed\grount_truth'
# # convertImageToBinary(path)
# anno(path)
