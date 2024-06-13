
#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash dist_train.sh DATA_PATH RANK_SIZE"
echo "For example: bash dist_train.sh /path/dataset 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

CKPT_PATH=$1

rm -rf $CKPT_PATH
mkdir $CKPT_PATH

mpirun -n 8 --allow-run-as-root python ./train.py --train_url $CKPT_PATH > $CKPT_PATH/train.log 2>&1 &
