#!/bin/bash

VERSION="d1"

DOWNLOAD_DIR="pretrained_models"

URL="http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_${VERSION}_coco17_tpu-32.tar.gz"

function validate_url() {
  if [[ $(wget -S --spider $1 2>&1 | grep 'HTTP/1.1 200 OK') ]]; then echo "true"; fi
}

if validate_url $URL; then
  # Do something when exists
  wget -nc -P $DOWNLOAD_DIR $URL
else
  # Return or print some error
  echo "does not exist"
fi

tar zxvf "${DOWNLOAD_DIR}/efficientdet_${VERSION}_coco17_tpu-32.tar.gz" -C ${DOWNLOAD_DIR}
