import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_seq2seq')
CKPT = 'weights/vgg_seq2seq_0.0824_0.1857_0.1141.pth'
config['weights'] = CKPT

config['cnn']['pretrained'] = True
config['device'] = 'cuda:0'
config['predictor']['beamsearch'] = False

detector = Predictor(config)


def get_ocr_results(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    s = detector.predict(image)
    return s
