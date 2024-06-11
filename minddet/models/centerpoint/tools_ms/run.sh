#!/bin/bashDATA_PATH
echo "For example: bash run.sh /path/dataset 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
EXEC_PATH=$(pwd)
export RANK_SIZE=$1
export RANK_START=$2
DEVICE_START=0
export RANK_TABLE_FILE=$EXEC_PATH/$3


test_dist_8pcs()
{
    export DEVICE_NUM=8
}

test_dist_2pcs()
{
    export DEVICE_NUM=2
}

test_dist_4pcs()
{
    export DEVICE_NUM=4
}

test_dist_16pcs()
{
    export DEVICE_NUM=8
}

test_dist_${RANK_SIZE}pcs

for((i=0;i<DEVICE_NUM;i++))
do
    export DEVICE_ID=$[i+DEVICE_START]
    export RANK_ID=$[i+RANK_START]
    echo "start training for device $RANK_ID"
    python3 -m tools_ms.train --gpus $RANK_SIZE > train.log$RANK_ID 2>&1 &
done
wait
