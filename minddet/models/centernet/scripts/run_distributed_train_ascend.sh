#!/bin/bash

echo "================================================================================================================"
echo "Please run the script as: "
echo "bash run_distributed_train_ascend.sh MINDRECORD_DIR RANK_TABLE_FILE LOAD_CHECKPOINT_PATH"
echo "for example: bash run_distributed_train_ascend.sh /path/mindrecord_dataset /path/hccl.json /path/load_ckpt"
echo "if no ckpt, just run: bash run_distributed_train_ascend.sh /path/mindrecord_dataset /path/hccl.json"
echo "It is better to use the absolute path."
echo "For hyper parameter, please note that you should customize the scripts:
          '{CUR_DIR}/scripts/ascend_distributed_launcher/hyper_parameter_config.ini' "
echo "================================================================================================================"
CUR_DIR=`pwd`
MINDRECORD_DIR=$1
HCCL_RANK_FILE=$2
CONFIG_FILE=$3
if [ $# == 4 ];
then
    LOAD_CHECKPOINT_PATH=$4
else
    LOAD_CHECKPOINT_PATH=""
fi


python ${CUR_DIR}/scripts/ascend_distributed_launcher/get_distribute_train_cmd.py \
    --run_script_dir=${CUR_DIR}/train.py \
    --hyper_parameter_config_dir=${CUR_DIR}/scripts/ascend_distributed_launcher/hyper_parameter_config.ini \
    --mindrecord_dir=$MINDRECORD_DIR \
    --load_checkpoint_path=$LOAD_CHECKPOINT_PATH \
    --hccl_config_dir=$HCCL_RANK_FILE \
    --hccl_time_out=1200 \
    --cmd_file=distributed_cmd.sh

bash distributed_cmd.sh
