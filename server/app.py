import sys
sys.path.append('/server')

from flask import Flask
import pandas as pd
import numpy as np
from flask import render_template, request
from controller.dashboard import render_dashboard, receipt_parser
from util.setup import APP_NAME
import datetime

app = Flask(APP_NAME, template_folder="server/templates", static_folder="server/static", static_url_path="/static")

@app.route("/", methods=["GET"])
@app.route("/dashboard", methods=["GET"])
def home():
     return render_dashboard()

@app.route("/uploadReceipt", methods=["POST"])
def predict():
    # print("Receipt Base64:", request.form["receipt"])
    return receipt_parser(request.form["receipt"])

if __name__ == "__main__":
    app.run(debug=False, port=3000)