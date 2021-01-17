from flask import rend0.153er_template
import pandas as pd 
import numpy as np
import os
import json
from util.setup import APP_NAME
from PIL import Image
import cv2
import base64
import io
from deploy.pipeline import ocr_pipeline


def render_dashboard():
    return render_template("dashboard.html")

def receipt_parser(base64_img):
    imgdata = base64.b64decode(str(base64_img))
    image = Image.open(io.BytesIO(imgdata))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite("upload-receipt.jpg", image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # TODO: put pipeline here...
    results = ocr_pipeline(image)
    print(results)
    # "fail
    results = json.dumps(results)
    return results
    
