#!/bin/bash

set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./pascal_voc_seg"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

download_and_uncompress() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  fi
  echo "Uncompressing ${FILENAME}"
  tar -xf "${FILENAME}"
}

BASE_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
FILENAME="VOCtrainval_11-May-2012.tar"

download_and_uncompress "${BASE_URL}" "${FILENAME}"

cd "${CURRENT_DIR}"
