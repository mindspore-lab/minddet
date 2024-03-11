#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_train_ascend.sh DEVICE_ID MINDRECORD_DIR LOAD_CHECKPOINT_PATH"
echo "for example: bash run_standalone_train_ascend.sh 0 /path/mindrecord_dataset /path/load_ckpt"
echo "if no ckpt, just run: bash run_standalone_train_ascend.sh 0 /path/mindrecord_dataset"
echo "=============================================================================================================="

DEVICE_ID=$1
MINDRECORD_DIR=$2
if [ $# == 3 ];
then
    LOAD_CHECKPOINT_PATH=$3
else
    LOAD_CHECKPOINT_PATH=""
fi

mkdir -p ms_log 
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
export DEVICE_ID=$DEVICE_ID

python ${PROJECT_DIR}/../train.py  \
    --distribute=false \
    --need_profiler=false \
    --profiler_path=./profiler \
    --device_id=$DEVICE_ID \
    --enable_save_ckpt=true \
    --do_shuffle=true \
    --enable_data_sink=true \
    --data_sink_steps=-1 \
    --epoch_size=330 \
    --load_checkpoint_path=$LOAD_CHECKPOINT_PATH \
    --save_checkpoint_steps=3664 \
    --save_checkpoint_num=1 \
    --mindrecord_dir=$MINDRECORD_DIR \
    --mindrecord_prefix="coco_det.train.mind" \
    --visual_image=false \
    --save_result_dir="" > training_log.txt 2>&1 &