#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash convert_dataset_to_mindrecord.sh /path/coco_dataset_dir /path/mindrecord_dataset_dir"
echo "=============================================================================================================="

COCO_DIR=$1
MINDRECORD_DIR=$2

export GLOG_v=1
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)

python ${PROJECT_DIR}/../src/dataset.py  \
    --coco_data_dir=$COCO_DIR \
    --mindrecord_dir=$MINDRECORD_DIR \
    --mindrecord_prefix="coco_det.train.mind" > create_dataset.log 2>&1 &