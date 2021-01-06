# from numba import cuda
#
# cuda.select_device(0)
# cuda.close()

import tensorflow as tf

from multiprocessing import Pool


def _process(image):
    sess = tf.Session()
    sess.close()


def process_image(image):
    with Pool(1) as p:
        return p.apply(_process, (image,))
