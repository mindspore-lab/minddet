
#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DATA_PATH RANK_SIZE"
echo "For example: bash run.sh /path/dataset 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="




export RANK_TABLE_FILE=$1
DATA_PATH=$2  #/root/data/data/caddn_data/data/kitti/
CKPT_PATH=$3
export RANK_SIZE=8
export STAR_DEVICE=0

rm -rf $CKPT_PATH
mkdir $CKPT_PATH
for((i=0;i<${RANK_SIZE};i++))
do
    export DEVICE_ID=$[i+STAR_DEVICE]
    export RANK_ID=$i
    rm -rf $CKPT_PATH/device$DEVICE_ID
    mkdir $CKPT_PATH/device$DEVICE_ID
    echo "start training for device $i"
    env > $CKPT_PATH/device$DEVICE_ID/env.log
    python3 ./train.py --data_url $DATA_PATH --train_url $CKPT_PATH > $CKPT_PATH/device$DEVICE_ID/train.log 2>&1 &
done
