#!/bin/bash

VERSION="d7"

DOWNLOAD_DIR="weights"

URL="http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_${VERSION}_coco17_tpu-32.tar.gz"

function validate_url() {
  if [[ $(wget -S --spider $1 2>&1 | grep 'HTTP/1.1 200 OK') ]]; then echo "true"; fi
}

if $(validate_url ${URL}) ; then wget -nc -P $DOWNLOAD_DIR $URL ; else echo "does not exist"; fi


tar zxvf "${DOWNLOAD_DIR}/efficientdet_${VERSION}_coco17_tpu-32.tar.gz" -C ${DOWNLOAD_DIR}
