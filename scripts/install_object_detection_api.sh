#!/bin/bash

cd libs

git clone --recursive https://github.com/tensorflow/models.git

cd models/research || exit

protoc object_detection/protos/*.proto --python_out=.

cp object_detection/packages/tf2/setup.py .

python -m pip install --use-feature=2020-resolver .
