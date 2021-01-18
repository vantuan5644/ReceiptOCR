import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

USE_DIFFERENT_OCR_MODELS = False

if not USE_DIFFERENT_OCR_MODELS:
    config = Cfg.load_config_from_name('vgg_seq2seq')
    # CKPT = 'weights/vgg_seq2seq_10k_finetuned_0.0909_0.1642_0.117.pth'
    CKPT = 'weights/vgg_seq2seq_10k_finetuned_0.0_0.0294_0.0.pth'
    config['weights'] = CKPT

    config['cnn']['pretrained'] = True
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch'] = False

    detector = Predictor(config)
else:
    configs = {'multiple_lines': Cfg.load_config_from_file('config_vgg_seq2seq_multiple_lines.yml'),
               'single_line': Cfg.load_config_from_file('config_vgg_seq2seq_single_line.yml')}
    detector_single_line = Predictor(configs['single_line'])
    detector_multiple_lines = Predictor(configs['multiple_lines'])

def get_ocr_results(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if USE_DIFFERENT_OCR_MODELS:
        results = {'single_line': detector_single_line.predict(image),
                   'multiple_lines': detector_multiple_lines.predict(image)}
    else:
        results = {'single_line': detector.predict(image),
                   'multiple_lines': detector.predict(image)}

        # print(results)
    return results
